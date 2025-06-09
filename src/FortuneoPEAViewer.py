import io
import streamlit as st
import pandas as pd
import numpy as np
from datetime import  date
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objs as go
import plotly.colors

st.set_page_config(layout="wide")

st.title("Suivi de PEA - Valorisation & Rendements")

st.sidebar.header("Import des fichiers")
titres_file = st.sidebar.file_uploader("Historique titres", type="csv")
especes_file = st.sidebar.file_uploader("Historique espèces", type="csv")
ticker_mapping_file = st.sidebar.file_uploader("Correspondance Tickers", type="csv")
# if especes_file is not None and especes_file != '':
#     ticker_mapping_file = os.path.dirname(titres_file) + "/tickerMatching.csv"

ticker_map = {}
if ticker_mapping_file:
    try:
        mapping_df = pd.read_csv(ticker_mapping_file, encoding="latin1", sep=";")
        ticker_map = dict(zip(mapping_df['label'], mapping_df['ticker']))
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier de correspondance : {e}")

if titres_file and especes_file:
    try:
        df_titres = pd.read_csv(titres_file, encoding="latin1", sep=";")
        df_especes = pd.read_csv(especes_file, encoding="latin1", sep=";")
    except Exception as e:
        st.error(f"Erreur de lecture des fichiers CSV : {e}")
        st.stop()

    try:
        df_titres["Date"] = pd.to_datetime(df_titres["Date"], dayfirst=True, errors='coerce')
        df_especes["Date opération"] = pd.to_datetime(df_especes["Date opération"], dayfirst=True, errors='coerce')
        df_especes["Date valeur"] = pd.to_datetime(df_especes["Date valeur"], dayfirst=True, errors='coerce')

        df_especes["Débit"] = pd.to_numeric(df_especes["Débit"].str.replace(",", "."), errors='coerce').fillna(0)
        df_especes["Crédit"] = pd.to_numeric(df_especes["Crédit"].str.replace(",", "."), errors='coerce').fillna(0)
    except Exception as e:
        st.error(f"Erreur lors du traitement des colonnes de dates ou de montants : {e}")
        st.stop()

    # Sélection de l'intervalle de temps
    all_dates = pd.concat([df_titres["Date"], df_especes["Date opération"], df_especes["Date valeur"]]).dropna()
    if all_dates.empty:
        st.error("Impossible de déterminer une plage de dates valide.")
        st.stop()

    # ticker_mapping_file = os.path.dirname(titres_file) + "/tickerMatching.csv"

    # Charger le mapping existant ou le créer si vide
    # try:
    #     mapping_df = pd.read_csv(ticker_mapping_file, encoding="latin1", sep=";")
    # except Exception:
    mapping_df = pd.DataFrame(columns=["label", "ticker"])        

    min_date = all_dates.min()
    max_date = date.today()
    start_date, end_date = st.sidebar.date_input("Période à afficher", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    pas_temps = st.sidebar.selectbox("Intervale", ["Jour", "Semaine", "Mois", "Année"])

    # Traitement des versements
    try:
        df_versements = df_especes[df_especes["libellé"].str.lower().str.contains("versement", na=False)]
        df_versements = df_versements[["Date opération", "Crédit"]].dropna()
        df_versements = df_versements.rename(columns={"Date opération": "Date", "Crédit": "Montant net"})

        # df_titres_sans_double_dividende = df_titres[df_titres["Opération"].str.contains("Achat|Vente|SCRIPT|RACHAT|Encaissement|TAXE", na=False)].copy()
        df_titres = df_titres[~df_titres["Opération"].str.contains("OST", na=False)].copy() # everything but OST (détachement de dividende)
        df_montantsNet = df_titres[["Date", "Montant net"]].dropna()
        df_allOperations = pd.concat([df_versements, df_montantsNet], axis=0, ignore_index=True)

        df_allOperations = df_allOperations.groupby("Date")["Montant net"].sum()
        df_compte_espece = df_allOperations.cumsum()

        # make df_versement cumulative
        df_versements = df_versements.groupby("Date")["Montant net"].sum().cumsum()

    except Exception as e:
        st.error(f"Erreur lors du traitement des versements : {e}")
        st.stop()

    # Traitement des positions journalières
    try:
        df_positions = df_titres[df_titres["Opération"].str.contains("Achat|Vente|SCRIPT|RACHAT", na=False)].copy()
        df_positions["Qté"] = pd.to_numeric(df_positions["Qté"], errors='coerce')
        df_positions["Qté"] = df_positions.apply(lambda row: row["Qté"] if 'Achat' in row["Opération"] or 'SCRIPT' in row["Opération"]  else -row["Qté"], axis=1)
        # df_positions = df_positions.dropna(subset=["Date", "Qté", "libellé"])

        theFreq='W'
        if pas_temps == "Jour":
            theFreq='B' # Business days
        elif pas_temps == "Mois":
            theFreq='BME' # Business Month End
        elif pas_temps == "Année":
            theFreq='BYE' # Business Year End
        
        dates_range = pd.date_range(start=start_date, end=end_date, freq=theFreq)
        titres = df_positions["libellé"].unique()

        position_matrix = pd.DataFrame(index=dates_range, columns=titres).astype(float).fillna(0)

        for _, row in df_positions.iterrows():
            position_matrix.loc[row["Date"]:, row["libellé"]] += row["Qté"]

    except Exception as e:
        st.error(f"Erreur lors de la reconstruction des positions journalières : {e}")
        st.stop()

    # Valorisation des titres via Yahoo Finance
    price_data = {}
    hist_data = {}
    valuation_details = []
    errors = []

    theInterval='1d'
    if pas_temps == "Semaine":
        theInterval='5d'
    elif pas_temps == "Mois":
        theInterval='1mo'
    elif pas_temps == "Année":
        theInterval='1y'    

    tickersInfo = dict()

    # st.sidebar.subheader("Valorisation actuelle estimée")
    for titre in position_matrix.columns:
        try:
            if ticker_map.get(titre) is not None:
                ticker = ticker_map[titre]
            else:            
                lookupTickers = yf.Lookup(titre).all
                if len(lookupTickers) > 0:
                    # favorise les tickers de paris, sur Fortuneo, il y a de bonnes chances que ce soit les bons
                    tickers_pa = [idx for idx in lookupTickers.index if idx.endswith('.PA')]
                    if tickers_pa:
                        ticker = tickers_pa[0]
                    else:
                        ticker = lookupTickers.index[0]

                    ticker_map[titre] = ticker
                else:
                    raise Exception(f'Introuvable sur yfinance, ajouter la correspondance ticker-titre dans le fichier tickerMatching.csv')

            hist = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=theInterval)#.ffill().bfill()
            hist.index = hist.index.tz_localize(None) # make dates compatible
            
            tickersInfo[titre] = yf.Ticker(ticker).info

            last_price = hist["Close"].iloc[-1]
            latest_qty = position_matrix[titre].iloc[-1]
            val = last_price * latest_qty

            price_data[titre] = position_matrix[titre].copy()
            hist_data[titre] = pd.Series(index=price_data[titre].index, dtype=float).fillna(0)
            for i in range (len(price_data[titre])):
                if pd.isna(price_data[titre].iloc[i]):
                    price_data[titre].iloc[i] = 0
                    hist_data[titre].iloc[i] = 0
                else:
                    aDate = price_data[titre].index[i]
                    previousHistIndex = hist.index.get_indexer([aDate], method='ffill')                        
                    price_data[titre].iloc[i] = price_data[titre].iloc[i] * hist["Close"].iloc[previousHistIndex] 
                    hist_data[titre].iloc[i] = hist["Close"].iloc[previousHistIndex]

            valuation_details.append({"Titre": titre, "Quantité": latest_qty, "Prix actuel": last_price, "Valeur": val})
        except Exception as e:
            st.sidebar.error(f'Erreur sur {titre} : {e}')
            print(f'Erreur sur {titre} : {e}')        
            errors.append(titre)
    df_price_total = pd.DataFrame(price_data)
    df_price_total["Total"] = df_price_total.sum(axis=1)
    df_valo = df_price_total["Total"]#.rename(columns={"Total": "Valorisation cumulée"})
    df_valo.index.name = "Date"

    st.subheader("Correspondance titres ↔ tickers Yahoo Finance")

    # Ajouter les titres manquants (présents dans le portefeuille mais pas dans le mapping)
    titres_portefeuille = sorted(set(df_titres["libellé"].unique()))
    for titre in titres_portefeuille:
        if titre not in mapping_df["label"].values:
            if titre in ticker_map:
                mapping_df = pd.concat([mapping_df, pd.DataFrame([{"label": titre, "ticker": ticker_map[titre]}])], ignore_index=True)
            else:   
                mapping_df = pd.concat([mapping_df, pd.DataFrame([{"label": titre, "ticker": ""}])], ignore_index=True)                

    # Affichage et édition du tableau
    edited_mapping = st.data_editor(
        mapping_df.sort_values("label").reset_index(drop=True),
        num_rows="dynamic",
        use_container_width=True,
        key="ticker_editor"
    )

    # Ajout du bouton de téléchargement, même si le fichier n'existe pas encore
    csv_buffer = io.StringIO()
    edited_mapping.to_csv(csv_buffer, encoding="latin1", sep=";", index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Télécharger la correspondance tickers (CSV)",
        data=csv_data,
        file_name="tickerMatching.csv",
        mime="text/csv"
    )

    if st.button("💾 Sauvegarder la correspondance tickers"):
        try:
            edited_mapping.drop_duplicates(subset=["label"], keep="last", inplace=True)
            edited_mapping.to_csv(ticker_mapping_file, encoding="latin1", sep=";", index=False)
            st.success("Fichier de correspondance sauvegardé ! Relancez l’application pour prise en compte.")
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde : {e}")

    st.markdown("---")    

    df_compte_espece.index = pd.to_datetime(df_compte_espece.index).tz_localize(None)
    # df_compte_especeNotReIndexed = df_compte_espece.copy()
    df_compte_espece = df_compte_espece.reindex(df_valo.index, method="ffill")

    df_valo = df_valo.add(df_compte_espece)

    df_perf = pd.concat([df_valo, df_compte_espece, df_versements], axis=1).ffill()
    df_perf.columns = ["Valorisation cumulée", "Compte Espèce", "Versements cumulés"]

    # S'assurer que l'index est bien un DatetimeIndex
    df_price_total.index = pd.to_datetime(df_price_total.index)

    if pas_temps == "Semaine":
        df_perf = df_perf.resample("W").last().ffill()
        df_price_total = df_price_total.resample("W").last().ffill()
    elif pas_temps == "Mois":
        df_perf = df_perf.resample("M").last().ffill()
        df_price_total = df_price_total.resample("M").last().ffill()
    elif pas_temps == "Année":
        df_perf = df_perf.resample("Y").last().ffill()
        df_price_total = df_price_total.resample("Y").last().ffill()
    else:
        df_perf = df_perf.resample("D").last().ffill()
        df_price_total = df_price_total.resample("D").last().ffill()

    df_perf = df_perf.ffill().bfill()
    df_perf["Flux"] = df_perf["Versements cumulés"].diff().fillna(0)
    element = df_perf["Versements cumulés"].loc[df_perf.index[0]]
    df_perf.loc[df_perf.index[0], 'Flux'] = element
    df_perf["Valeur"] = df_perf["Valorisation cumulée"]

    # --- Ajout graphique rendement global ---
    rendement_global = (df_perf["Valorisation cumulée"] / df_perf["Versements cumulés"] - 1).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

    # --- Courbes de valorisation et versements ---
    st.subheader("Courbes de valorisation et versements")
    fig_val = go.Figure()
    for col in ["Valorisation cumulée", "Compte Espèce", "Versements cumulés"]:
        fig_val.add_trace(go.Scatter(
            x=df_perf.index,
            y=df_perf[col],
            mode='lines',
            name=col,
            hovertemplate=f"{col}<br>Valeur: %{{y:,.2f}} €<extra></extra>"
        ))
    fig_val.update_layout(
        xaxis_title="Date",
        yaxis_title="Montant (€)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # --- Rendement global ---
    st.subheader("Rendement global ( (Valorisation +  especes) / Versements cumulés - 1)")
    fig_rend = go.Figure()
    fig_rend.add_trace(go.Scatter(
        x=df_perf.index,
        y=rendement_global,
        mode='lines',
        name="Rendement global",
        hovertemplate=f"Rendement global<br>%{{y:,.2f}}%<extra></extra>"
    ))
    fig_rend.update_layout(
        xaxis_title="Date",
        yaxis_title="Rendement (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_rend, use_container_width=True)
    
    # st.subheader("Évolution de chaque position depuis le premier achat (base 0% à l'achat initial)")
    evolution_positions = pd.DataFrame(index=df_price_total.index)
    
    for titre in price_data:
        # Trouver la première date d'achat
        mask = (df_titres["libellé"] == titre) & (df_titres["Opération"].str.contains("Achat|SCRIPT", na=False))
        achats = df_titres[mask][["Date", "Montant net"]].dropna()
        if achats.empty:
            evolution_positions[titre] = 0
            continue
        first_invest_date = achats["Date"].min()
        histData = hist_data[titre]
        try:
            first_invest_loc = histData.index[histData.index >= first_invest_date][0]
            base_value = histData.loc[first_invest_loc]
            evolution = pd.Series(0, index=histData.index, dtype=float)
            qty = position_matrix[titre].reindex(histData.index).fillna(0)

            evolution.loc[:] = (histData.loc[:] / base_value -1)*100

            evolution = evolution.reindex(df_price_total.index).ffill().fillna(0)
            evolution_positions[titre] = evolution
        except IndexError:
            evolution_positions[titre] = 0 

    
    fig = go.Figure()
    # Assign a color to each title
    palette = plotly.colors.qualitative.Vivid
    color_map = {titre: palette[i % len(palette)] for i, titre in enumerate(evolution_positions.columns)}

    for titre in evolution_positions.columns:
        hist_data[titre] = hist_data[titre].reindex(df_price_total.index).ffill().fillna(0)
        qty = position_matrix[titre].reindex(df_price_total.index).ffill().fillna(0)
        evolution = evolution_positions[titre]
        color = color_map[titre]

        holding = qty > 0
        if holding.any():
            first_hold = holding.idxmax()
            after_first = holding.loc[first_hold:]
            if (~after_first).any():
                first_zero = after_first[~after_first].index[0]
            else:
                first_zero = evolution.index[-1]

            profitMarginsValue = tickersInfo[titre]['profitMargins'] if 'profitMargins' in tickersInfo[titre] else 'NA'
            dividendRateValue = tickersInfo[titre]['dividendRate'] if 'dividendRate' in tickersInfo[titre] else 'NA'
            traillingPEValue = tickersInfo[titre]['trailingPE'] if 'trailingPE' in tickersInfo[titre] else 'NA'
            forwardPEValue = tickersInfo[titre]['forwardPE'] if 'forwardPE' in tickersInfo[titre] else 'NA'
            priceToBookValue = tickersInfo[titre]['priceToBook'] if 'priceToBook' in tickersInfo[titre] else 'NA'
        
            # Dotted line: before first hold or from first_zero to the end
            fig.add_trace(go.Scatter(
                x=evolution.index,
                y=[e if (d <= first_hold or d >= first_zero) else None for d, e in zip(evolution.index, evolution)],
                mode='lines',
                name=titre,
                line=dict(dash='dot', color=color),
                legendgroup=titre,
                text=[f"{titre}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évolution: {evolution.iloc[i]:.2f}%<br>Position: {df_price_total[titre].iloc[i]:,.2f} €<br>profitMargins:{profitMarginsValue}<br>Dividend Rate: {dividendRateValue}<br>Trailing P/E: {traillingPEValue}<br>Forward P/E: {forwardPEValue}<br>Price to Book: {priceToBookValue}"
                    for i in range(len(evolution))],
                hoverinfo='text'
            ))
            # Solid line: from first_hold to first_zero (excluded)
            fig.add_trace(go.Scatter(
                x=evolution.index,
                y=[e if (d >= first_hold and d <= first_zero) else None for d, e in zip(evolution.index, evolution)],
                mode='lines',
                name=titre,
                line=dict(dash='solid', color=color),
                legendgroup=titre,
                showlegend=False,
                text=[f"{titre}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évolution: {evolution.iloc[i]:.2f}%<br>Position: {df_price_total[titre].iloc[i]:,.2f} €<br>profitMargins:{profitMarginsValue}<br>Dividend Rate: {dividendRateValue}<br>Trailing P/E: {traillingPEValue}<br>Forward P/E: {forwardPEValue}<br>Price to Book: {priceToBookValue}"
                    for i in range(len(evolution))],
                hoverinfo='text'
            ))                
            # Ajout des marqueurs d'achats et ventes
            df_ops = df_titres[df_titres["libellé"] == titre]
            for op_type, color_marker, symbol in [
                ("Achat", "green", "triangle-up"),
                ("SCRIPT", "green", "triangle-up"),                
                ("Vente", "red", "triangle-down"),
                ("RACHAT", "red", "triangle-down")                
            ]:
                ops = df_ops[df_ops["Opération"].str.contains(op_type, na=False)]
                if not ops.empty:
                    ops = ops.groupby("Date", as_index=False).agg({"Montant net": "sum"})
                    dates_ops = ops["Date"]
                    y_ops = evolution.loc[dates_ops].values
                    montant_ops = -ops["Montant net"].values
                    fig.add_trace(go.Scatter(
                        x=dates_ops,
                        y=y_ops,
                        mode="markers",
                        name=f"{titre} {op_type}",
                        marker=dict(color=color_marker, symbol=symbol, size=10, line=dict(width=1, color='black')),
                        legendgroup=titre,
                        showlegend=False,
                        hovertemplate=(
                            f"{titre} - {op_type}<br>"
                            "Date: %{x|%Y-%m-%d}<br>"
                            "Évolution: %{y:.2f}%<br>"
                            "Montant: %{customdata:,.2f} €<extra></extra>"
                        ),
                        customdata=montant_ops.reshape(-1, 1)
                    ))
        else:
            continue


    fig.update_layout(
        title="Évolution de chaque position (base 0% au premier achat)",
        xaxis_title="Date",
        yaxis_title="Évolution (%)",
        hovermode="closest" # "x" for boxes or "x unified" for all
    )

    st.plotly_chart(fig, use_container_width=True)
        # st.line_chart(evolution_positions)
    
    # valorisation cumulée  + compte espece
    current_valo = df_valo.iloc[-1] + df_compte_espece.iloc[-1]
    st.sidebar.markdown(f"**Valorisation estimée actuelle :** {current_valo:,.2f} €")

    st.subheader("Valorisation par ligne")
    fig_perf = go.Figure()
    for titre in df_price_total.drop(columns="Total").columns:
        fig_perf.add_trace(go.Scatter(
            x=df_price_total.index,
            y=df_price_total[titre],
            mode='lines',
            name=titre,
            hovertemplate=f"{titre}<br>Valeur: %{{y:,.2f}} €<extra></extra>"
        ))
    fig_perf.update_layout(
        xaxis_title="Date",
        yaxis_title="Valorisation (€)",
        hovermode="closest"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Comparaison d'une action avec un indice")

    # 1. Sélecteurs
    liste_indices = {
        "MSCI World": "^990100-USD-STRD",
        "CAC40": "^FCHI",
        "S&P500": "^GSPC",
        "NASDAQ100": "^NDX"
    }
    titre_choisi = st.selectbox("Choisir une action du portefeuille", list(df_price_total.drop(columns="Total").columns))
    indice_choisi_nom = st.selectbox("Choisir un indice de référence", list(liste_indices.keys()))
    indice_choisi_ticker = liste_indices[indice_choisi_nom]

    convert_fx = st.checkbox("Intégrer le taux de change EUR/USD pour les indices internationaux", value=True)

    # 2. Récupération des données
    first_invest_date = df_titres[(df_titres["libellé"] == titre_choisi) & (df_titres["Opération"].str.contains("Achat|SCRIPT", na=False))]["Date"].min()
    if pd.isna(first_invest_date):
        st.warning("Achat introuvable pour ce titre.")
    else:
        # Récupère les prix de l'action et de l'indice
        prix_action = hist_data[titre_choisi].reindex(df_price_total.index).ffill()
        hist_indice = yf.Ticker(indice_choisi_ticker).history(start=start_date, end=end_date, interval=theInterval) #start=first_invest_date, end=df_price_total.index[-1]
        hist_indice.index = hist_indice.index.tz_localize(None)
        prix_indice = hist_indice["Close"].reindex(df_price_total.index).ffill()

        if convert_fx and indice_choisi_nom in ["MSCI World", "S&P500", "NASDAQ100"]:
            # Récupère le taux EUR/USD sur la même période
            eurusd = yf.Ticker("EURUSD=X").history(start=start_date, end=end_date, interval=theInterval)
            eurusd.index = eurusd.index.tz_localize(None)
            taux_eur_usd = eurusd["Close"].reindex(df_price_total.index).ffill()
            # Conversion USD -> EUR (1 USD = X EUR donc 1 EUR = 1/X USD)
            prix_indice = prix_indice / taux_eur_usd        

        # 3. Calcul base 0% à la date du premier achat
        try:
            base_action = prix_action.loc[first_invest_date]
            base_indice = prix_indice.loc[first_invest_date]
            evol_action = (prix_action / base_action - 1) * 100
            evol_indice = (prix_indice / base_indice - 1) * 100
        except Exception:
            st.warning("Impossible de calculer l'évolution (données manquantes).")
            evol_action = evol_indice = pd.Series(index=prix_action.index, data=np.nan)

    # 4. Affichage graphique
    fig_compare = go.Figure()

    # Action : dot avant achat, solide après
    fig_compare.add_trace(go.Scatter(
        x=evol_action.index,
        y=[evol_action.loc[d] if d <= first_invest_date else None for d in evol_action.index],
        mode='lines',
        name=f"{titre_choisi} (avant achat)",
        line=dict(color='blue', dash='dot'),
        legendgroup=titre_choisi,
        showlegend=False,
        hovertemplate=f"{titre_choisi}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
    ))
    fig_compare.add_trace(go.Scatter(
        x=evol_action.index,
        y=[evol_action.loc[d] if d >= first_invest_date else None for d in evol_action.index],
        mode='lines',
        name=titre_choisi,
        legendgroup=titre_choisi,
        line=dict(color='blue', dash='solid'),
        hovertemplate=f"{titre_choisi}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
    ))

    # Indice : dot avant achat, solide après
    fig_compare.add_trace(go.Scatter(
        x=evol_indice.index,
        y=[evol_indice.loc[d] if d <= first_invest_date else None for d in evol_indice.index],
        mode='lines',
        name=f"{indice_choisi_nom} (avant achat)",
        line=dict(color='orange', dash='dot'),
        showlegend=False,
        legendgroup=indice_choisi_nom,
        hovertemplate=f"{indice_choisi_nom}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
    ))
    fig_compare.add_trace(go.Scatter(
        x=evol_indice.index,
        y=[evol_indice.loc[d] if d >= first_invest_date else None for d in evol_indice.index],
        mode='lines',
        name=indice_choisi_nom,
        legendgroup=indice_choisi_nom,
        line=dict(color='orange', dash='solid'),
        hovertemplate=f"{indice_choisi_nom}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
    ))

    fig_compare.update_layout(
        title=f"Comparaison {titre_choisi} vs {indice_choisi_nom} (base 0% au premier achat)",
        xaxis_title="Date",
        yaxis_title="Évolution (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_compare, use_container_width=True)  

    if errors:
        st.sidebar.warning("Titres non valorisés (erreurs ou manque de données Yahoo):")
else:
    st.info("Veuillez importer les deux fichiers pour commencer.")
