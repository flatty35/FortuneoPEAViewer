import sys
from tempfile import NamedTemporaryFile
# import io
from yfinance import Lookup, Ticker
import pandas as pd
import numpy as np
from datetime import date
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QDateEdit, QCheckBox, QDockWidget, QLineEdit, QToolButton, QGridLayout, QSizePolicy,
    QScrollArea, QHeaderView
)
from PySide6.QtCore import Qt, QDate, QSettings, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from plotly.colors import qualitative
from plotly.graph_objs import Figure, Scatter

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suivi de PEA - Valorisation & Rendements")
        self.resize(1400, 900)
        self.df_titres = None
        self.df_especes = None
        self.mapping_df = pd.DataFrame(columns=["label", "ticker"])
        self.ticker_map = {}
        self.settings = QSettings("FortuneoPEAViewer", "UserFiles")
        
        # for comparison update
        self.df_titres_modified = None
        self.df_price_total = None
        self.hist_data = None
        self.start_date = None
        self.end_date = None
        self.theInterval = None

        self.sectionsHTML = dict()
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.tickersRecallLayout = QHBoxLayout()
        self.btn_show_dock = QPushButton("Tickers")
        self.btn_show_dock.setVisible(False)
        self.btn_show_dock.clicked.connect(self.restore_dock)        

        # --- Import group with persistent file paths ---
        import_group = QGroupBox("") #Import des fichiers
        import_layout = QGridLayout()
        
        self.file_fields = {}
        for label, key, loader in [
            ("Historique titres", "titres_file", self.load_titres),
            ("Historique espèces", "especes_file", self.load_especes),
            ("Correspondance Tickers", "ticker_mapping_file", self.load_mapping)
        ]:
            # row = QHBoxLayout()
            import_layout.addWidget(QLabel(label), len(self.file_fields), 0)
            # row.addWidget(QLabel(label))
            line_edit = QLineEdit()
            line_edit.setText(self.settings.value(key, ""))
            browse_btn = QPushButton("...")
            import_layout.addWidget(line_edit, len(self.file_fields), 1)
            import_layout.addWidget(browse_btn, len(self.file_fields), 2)
            # import_layout.addLayout(row)
            self.file_fields[key] = (line_edit, browse_btn)
            # Connexion du bouton browse
            browse_btn.clicked.connect(lambda checked, k=key, le=line_edit: self.browse_file(k, le, loader))

        import_group.setLayout(import_layout)

        # Ajout d'un bouton rétractable pour le groupe d'import

        import_toggle = QToolButton()
        import_toggle.setText("Import des fichiers")
        import_toggle.setCheckable(True)
        import_toggle.setChecked(True)
        import_toggle.setArrowType(Qt.DownArrow)
        import_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # <-- Affiche texte + flèche
        import_toggle.clicked.connect(
            lambda checked: (
                import_group.setVisible(checked),
                import_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            )
        )

        self.tickersRecallLayout.addWidget(import_toggle)
        self.tickersRecallLayout.addStretch()
        self.tickersRecallLayout.addWidget(self.btn_show_dock)
        left_layout.addLayout(self.tickersRecallLayout)

        # left_layout.addWidget(import_toggle)
        left_layout.addWidget(import_group)

        options_group = QGroupBox("") #Import des fichiers
        options_layout = QVBoxLayout()

        # Date range selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Période à afficher :"))
        self.date_start = QDateEdit()
        self.date_end = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_end.setCalendarPopup(True)

        date_layout.addWidget(self.date_start)
        date_layout.addWidget(QLabel("→"))
        date_layout.addWidget(self.date_end)
        date_layout.addStretch()
        options_layout.addLayout(date_layout)

        # Interval selection
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Intervalle :"))
        self.combo_interval = QComboBox()
        self.combo_interval.addItems(["Jour", "Semaine", "Mois", "Année"])
        interval_layout.addWidget(self.combo_interval)
        interval_layout.addStretch()
        options_layout.addLayout(interval_layout)

        # 1. Sélecteurs
        self.liste_indices = {
            "MSCI World": "^990100-USD-STRD",
            "CAC40": "^FCHI",
            "S&P500": "^GSPC",
            "NASDAQ100": "^NDX"
        }

        # Met à jour la liste des indices
        indices = list(self.liste_indices.keys())

        options_group.setLayout(options_layout)

        # Ajout d'un bouton rétractable pour le groupe d'options
        options_toggle = QToolButton()
        options_toggle.setText("Options")
        options_toggle.setCheckable(True)
        options_toggle.setChecked(True)
        options_toggle.setArrowType(Qt.DownArrow)
        options_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)  # <-- Affiche texte + flèche
        options_toggle.clicked.connect(
            lambda checked: (
                options_group.setVisible(checked),
                options_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            )
        )

        left_layout.addWidget(options_toggle)
        left_layout.addWidget(options_group)        

        plotly_container = QWidget()
        plotly_layout = QVBoxLayout(plotly_container)
        self.section_combo = QComboBox()
        self.section_combo.setPlaceholderText("Choisir une section à afficher")
        self.section_combo.currentIndexChanged.connect(self.on_section_changed)

        # Sélecteur d'action du portefeuille
        self.compare_widget = QWidget()

        self.compare_section_name = "Comparaison d'une action avec un indice"
        compare_widget_layout = QVBoxLayout()
        # Checkbox for FX
        self.checkbox_fx = QCheckBox("Intégrer le taux de change EUR/USD pour les indices internationaux")
        self.checkbox_fx.setChecked(True)

        compare_layout = QHBoxLayout()
        self.combo_titre = QComboBox()
        self.combo_titre.setPlaceholderText("Choisir une action du portefeuille à comparer")

        # Sélecteur d'indice de référence
        self.combo_indice = QComboBox()
        self.combo_indice.setPlaceholderText("Choisir un indice de référence à comparer")
        self.combo_indice.addItems(indices)

        self.combo_titre.currentIndexChanged.connect(self.updateComparison)
        self.combo_indice.currentIndexChanged.connect(self.updateComparison)        

        # options_layout.addWidget(QLabel("Indice de référence à comparer :"))
        compare_layout.addWidget(QLabel("Comparer : "))
        compare_layout.addWidget(self.combo_titre)
        compare_layout.addWidget(QLabel(" Avec "))
        compare_layout.addWidget(self.combo_indice)
        compare_layout.addStretch()
        compare_widget_layout.addLayout(compare_layout)
        compare_widget_layout.addWidget(self.checkbox_fx)

        self.compare_widget.setLayout(compare_widget_layout)
        self.compare_widget.hide()

        self.plotly_view = QWebEngineView()
        self.plotly_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plotly_layout.addWidget(self.section_combo)
        plotly_layout.addWidget(self.compare_widget)
        # plotly_layout.addLayout(compare_layout)

        # Création du widget contenant le QWebEngineView
        plotly_layout.addWidget(self.plotly_view)

        # Création de la scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(plotly_container)

        # Ajout au layout principal
        left_layout.addWidget(scroll_area, stretch=2)

        # Status label
        # self.status_group = QGroupBox("Status") #Import des fichiers
        # self.status_layout = QVBoxLayout()
        self.label_status = QLabel("Veuillez charger les fichiers.")
        # self.status_layout.addWidget(self.label_status)
        # self.status_group.setLayout(self.status_layout)
        left_layout.addWidget(self.label_status )

        main_layout.addLayout(left_layout)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # --- RIGHT SIDE ---
        group = QGroupBox("Correspondance titres ↔ tickers Yahoo Finance")
        group_layout = QVBoxLayout()
        self.table_mapping = QTableWidget(0, 2)
        self.table_mapping.setHorizontalHeaderLabels(["label", "ticker"])
        self.table_mapping.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)  # label prend le reste
        self.table_mapping.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)  # ticker s'ajuste

        group_layout.addWidget(self.table_mapping, stretch=1)
        self.btn_save_mapping = QPushButton("💾 Sauvegarder la correspondance tickers")
        group_layout.addWidget(self.btn_save_mapping)
        group.setLayout(group_layout)

        dock = QDockWidget("Correspondance tickers", self)
        dock.setWidget(group)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.visibilityChanged.connect(self.on_dock_visibility_changed)
        self.dock = dock  # garde une référence pour restore_dock        

        # Connections (inchangées)
        # self.btn_load_titres.clicked.connect(self.load_titres)
        # self.btn_load_especes.clicked.connect(self.load_especes)
        # self.btn_load_mapping.clicked.connect(self.load_mapping)
        self.btn_save_mapping.clicked.connect(self.save_mapping)
        self.combo_interval.currentIndexChanged.connect(self.process_and_plot)
        self.checkbox_fx.stateChanged.connect(self.updateComparison)
        self.date_start.dateChanged.connect(self.process_and_plot)
        self.date_end.dateChanged.connect(self.process_and_plot)

        # load everything
        self.load_titres()
        self.load_especes()
        self.load_mapping()
        self.process_and_plot()

    def on_dock_visibility_changed(self, visible):
        self.btn_show_dock.setVisible(not visible)

    def restore_dock(self):
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()        

    def browse_file(self, key, line_edit, loader):
        path, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier", "", "CSV Files (*.csv)")
        if path:
            line_edit.setText(path)
            self.settings.setValue(key, path)
            loader()

    # Pour charger les fichiers, utiliser self.file_fields['titres_file'][0].text() etc.
    def load_titres(self):
        path = self.file_fields['titres_file'][0].text()
        if path:
            try:
                self.df_titres = pd.read_csv(path, encoding="latin1", sep=";")
                self.label_status.setText("Historique titres chargé.")
                self.update_dates()

                # MAJ les titres dispo
                titres = list(set(self.df_titres["libellé"]))

                # Met à jour la liste des titres (actions du portefeuille)
                self.combo_titre.blockSignals(True)
                self.combo_titre.clear()
                self.combo_titre.addItems(titres)
                self.combo_titre.blockSignals(False)

                self.process_and_plot()


            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de lecture du fichier titres : {e}")

    def load_especes(self):
        path = self.file_fields['especes_file'][0].text()
        if path:
            try:
                self.df_especes = pd.read_csv(path, encoding="latin1", sep=";")
                self.label_status.setText("Historique espèces chargé.")
                self.update_dates()
                self.process_and_plot()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de lecture du fichier espèces : {e}")

    def update_dates(self):
        if self.df_titres is None or self.df_especes is None:
            return

        # Sélection de l'intervalle de temps
        all_dates = pd.concat([self.df_titres["Date"], self.df_especes["Date opération"], self.df_especes["Date valeur"]]).dropna()
        if all_dates.empty:
            self.label_status.setText(f"Impossible de déterminer une plage de dates valide.")
            return            

        min_date = pd.to_datetime(all_dates, dayfirst=True, errors='coerce').min()
        # min_date = datetime.strptime(min_date, "%d/%m/%Y")
        max_date = date.today()

        self.date_start.setDate(QDate(min_date.year, min_date.month, min_date.day))
        self.date_end.setDate(QDate(max_date.year, max_date.month, max_date.day))

    def load_mapping(self):
        path = self.file_fields['ticker_mapping_file'][0].text()
        if path:
            try:
                self.mapping_df = pd.read_csv(path, encoding="latin1", sep=";")
                self.update_mapping_table()
                self.label_status.setText("Correspondance tickers chargée.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de lecture du mapping : {e}")

    def update_mapping_table(self):
        self.table_mapping.setRowCount(len(self.mapping_df))
        for i, row in self.mapping_df.iterrows():
            self.table_mapping.setItem(i, 0, QTableWidgetItem(str(row["label"])))
            self.table_mapping.setItem(i, 1, QTableWidgetItem(str(row["ticker"])))

        self.table_mapping.resizeColumnToContents(1)

    def save_mapping(self):
        path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder la correspondance tickers", "tickerMatching.csv", "CSV Files (*.csv)")
        if path:
            try:
                data = []
                for row in range(self.table_mapping.rowCount()):
                    label = self.table_mapping.item(row, 0).text() if self.table_mapping.item(row, 0) else ""
                    ticker = self.table_mapping.item(row, 1).text() if self.table_mapping.item(row, 1) else ""
                    data.append({"label": label, "ticker": ticker})
                df = pd.DataFrame(data)
                df.to_csv(path, encoding="latin1", sep=";", index=False)
                self.label_status.setText("Mapping sauvegardé.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur lors de la sauvegarde : {e}")

    def addSection(self, title, fig):
        html = fig.to_html(full_html=True, include_plotlyjs='cdn')
        lenSection = len(html.encode("utf-8"))
        if lenSection > 2_000_000:
            # Write to a temp file
            tmp = NamedTemporaryFile(delete=False, suffix=".html")
            tmp.write(html.encode("utf-8"))
            tmp.close()
            # Store the file path or QUrl for this section
            self.sectionsHTML[title] = QUrl.fromLocalFile(tmp.name)
            self.label_status.setText(
                f"Section '{title}' is too large ({lenSection} bytes), loaded from disk."
            )
        else:
            self.sectionsHTML[title] = html
    
    def process_and_plot(self):
        # Vérifie que les deux fichiers sont chargés
        if self.df_titres is None or self.df_especes is None or len(self.mapping_df) == 0:
            self.label_status.setText("Veuillez charger les deux fichiers.")
            return

        # update values with QEdit        
        start_date  = self.date_start.date().toPython()
        end_date  = self.date_end.date().toPython()
        pas_temps = self.combo_interval.currentText()
        
        # align dates
        try:
            # Nettoyage et parsing des dates
            df_titres = self.df_titres.copy()
            df_especes = self.df_especes.copy()

            df_titres["Date"] = pd.to_datetime(df_titres["Date"], dayfirst=True, errors='coerce')
            df_especes["Date opération"] = pd.to_datetime(df_especes["Date opération"], dayfirst=True, errors='coerce')
            df_especes["Date valeur"] = pd.to_datetime(df_especes["Date valeur"], dayfirst=True, errors='coerce')

            df_especes["Débit"] = pd.to_numeric(df_especes["Débit"].str.replace(",", "."), errors='coerce').fillna(0)
            df_especes["Crédit"] = pd.to_numeric(df_especes["Crédit"].str.replace(",", "."), errors='coerce').fillna(0)
        except Exception as e:
            self.label_status.setText(f"Erreur lors du traitement des colonnes de dates ou de montants : {e}")            

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
            self.label_status.setText(f"Erreur lors du traitement des versements : {e}")            

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
            self.label_status.setText(f"Erreur lors de la reconstruction des positions journalières : {e}")            

        try:
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

            ticker_map = dict(zip(self.mapping_df["label"], self.mapping_df["ticker"]))
            # ticker_mapping_file = self.file_fields['ticker_mapping_file'][0].text()

            for titre in position_matrix.columns:
                try:
                    if ticker_map.get(titre) is not None:
                        ticker = ticker_map[titre]
                    else:            
                        lookupTickers = Lookup(titre).all
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

                    hist = Ticker(ticker).history(start=start_date, end=end_date, interval=theInterval)#.ffill().bfill()
                    hist.index = hist.index.tz_localize(None) # make dates compatible
                    
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

                    # aTicker = Ticker(ticker)
                    # tickersInfo[titre] = aTicker.info.copy()
                except Exception as e:
                    self.label_status.setText(f'Erreur sur {titre} : {e}')
                    print(f'Erreur sur {titre} : {e}')        
                    errors.append(titre)
            df_price_total = pd.DataFrame(price_data)
            df_price_total["Total"] = df_price_total.sum(axis=1)
            df_valo = df_price_total["Total"]#.rename(columns={"Total": "Valorisation cumulée"})
            df_valo.index.name = "Date"

            # update mapping_df after looking for missing tickers
            self.mapping_df = self.mapping_df.from_dict(ticker_map, orient='index', columns=['ticker']).reset_index().rename(columns={'index': 'label'})
            self.update_mapping_table()

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

            # valorisation cumulée  + compte espece
            current_valo = df_valo.iloc[-1] + df_compte_espece.iloc[-1]

            self.sectionsHTML['Valorisation estimée'] = '<html>\n<head><meta charset="utf-8" /></head>\n<body>\n'
            self.sectionsHTML['Valorisation estimée'] += f"<div style='font-size:1.3em;font-weight:bold;margin:30px 0 10px 0;'>**Valorisation estimée actuelle :** {current_valo:,.2f} €</div>"
            self.sectionsHTML['Valorisation estimée'] += "<hr style='margin:40px 0;'>"

            # --- Ajout graphique rendement global ---
            rendement_global = (df_perf["Valorisation cumulée"] / df_perf["Versements cumulés"] - 1).replace([np.inf, -np.inf], np.nan).fillna(0) * 100

            # --- Courbes de valorisation et versements ---
            
            fig_val = Figure()
            for col in ["Valorisation cumulée", "Compte Espèce", "Versements cumulés"]:
                fig_val.add_trace(Scatter(
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

            self.addSection("Courbes de valorisation et versements", fig_val)
        
            # --- Rendement global ---
            fig_rend = Figure()
            fig_rend.add_trace(Scatter(
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
            self.addSection("Rendement global ( (Valorisation +  especes) / Versements cumulés - 1)", fig_rend)

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

            
            fig = Figure()
            # Assign a color to each title
            palette = qualitative.Vivid
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

                    # profitMarginsValue = tickersInfo[titre]['profitMargins'] if 'profitMargins' in tickersInfo[titre] else 'NA'
                    # dividendRateValue = tickersInfo[titre]['dividendRate'] if 'dividendRate' in tickersInfo[titre] else 'NA'
                    # traillingPEValue = tickersInfo[titre]['trailingPE'] if 'trailingPE' in tickersInfo[titre] else 'NA'
                    # forwardPEValue = tickersInfo[titre]['forwardPE'] if 'forwardPE' in tickersInfo[titre] else 'NA'
                    # priceToBookValue = tickersInfo[titre]['priceToBook'] if 'priceToBook' in tickersInfo[titre] else 'NA'

                    # customText=[f"{titre}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évolution: {evolution.iloc[i]:.2f}%<br>Position: {df_price_total[titre].iloc[i]:,.2f} €<br>profitMargins:{profitMarginsValue}<br>Dividend Rate: {dividendRateValue}<br>Trailing P/E: {traillingPEValue}<br>Forward P/E: {forwardPEValue}<br>Price to Book: {priceToBookValue}" for i in range(len(evolution))]
                    # customText=[f"{titre}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évolution: {evolution.iloc[i]:.2f}%<br>Position: {df_price_total[titre].iloc[i]:,.2f} €" for i in range(len(evolution))],
                    # customText=[f"{titre}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évo: {evolution.iloc[i]:.2f}%<br>Pos.: {df_price_total[titre].iloc[i]:,.2f} €" for i in range(len(evolution))],
                    # customText="test"


                    # hoverTemplateFormula=[f"{name}<br>Date: {evolution.index[i].strftime('%Y-%m-%d')}<br>Cote: {hist_data[titre].iloc[i]:.2f}€<br>Évolution: {evolution.iloc[i]:.2f}%<br>Position: {df_price_total[titre].iloc[i]:,.2f} €" for i in range(len(evolution))],

                    # fig.add_trace(Scatter(
                    #     x=dates,
                    #     y=values,
                    #     customdata=short_labels,  # array of short strings or numbers
                    #     mode='lines+markers',
                    #     hovertemplate="Date: %{x}<br>Valeur: %{y:.2f} €<br>Label: %{customdata}<extra></extra>"
                    # ))
                                

                    df_triplet = pd.DataFrame({"hist": hist_data[titre],"evolution": evolution,"price_total": df_price_total[titre]})

                    hovertemplateString="Date: %{x}<br>Cote: %{customdata[0]:.2f}€<br>Évolution: %{customdata[1]:.2f}%<br>Position: %{customdata[2]:,.2f} €"
                    # Dotted line: before first hold or from first_zero to the end
                    fig.add_trace(Scatter(
                        x=evolution.index,
                        y=[e if (d <= first_hold or d >= first_zero) else None for d, e in zip(evolution.index, evolution)],
                        mode='lines',
                        name=titre,
                        customdata=df_triplet,
                        line=dict(dash='dot', color=color),
                        legendgroup=titre,
                        hoverinfo='text',
                        hovertemplate=hovertemplateString
                    ))

                    # Solid line: from first_hold to first_zero (excluded)
                    fig.add_trace(Scatter(
                        x=evolution.index,
                        y=[e if (d >= first_hold and d <= first_zero) else None for d, e in zip(evolution.index, evolution)],
                        mode='lines',
                        name=titre,
                        customdata=df_triplet,
                        line=dict(dash='solid', color=color),
                        legendgroup=titre,
                        showlegend=False,
                        # text=customText,
                        hoverinfo='text',
                        #x.strftime('%Y-%m-%d')
                        hovertemplate=hovertemplateString
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

                            idx = evolution.index.searchsorted(dates_ops, side='left')
                            idx = np.clip(idx, 0, len(evolution.index) - 1)
                            y_ops = evolution.values[idx]

                            # y_ops = evolution.loc[dates_ops].values
                            montant_ops = -ops["Montant net"].values
                            fig.add_trace(Scatter(
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

            self.addSection("Évolution de chaque position depuis le premier achat (base 0% à l'achat initial)", fig)

            fig_perf = Figure()
            for titre in df_price_total.drop(columns="Total").columns:
                fig_perf.add_trace(Scatter(
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

            self.addSection("Valorisation par ligne", fig_perf)
            
            self.df_titres_modified = df_titres
            self.df_price_total = df_price_total
            self.hist_data = hist_data
            self.start_date = start_date
            self.end_date = end_date
            self.theInterval = theInterval

            self.updateComparison()

            self.label_status.setText("Graphique mis à jour.")
        except Exception as e:
            self.label_status.setText(f"Erreur lors du traitement : {e}")

        self.section_combo.blockSignals(True)
        self.section_combo.clear()
        self.section_combo.addItems(list(self.sectionsHTML.keys()))
        self.section_combo.blockSignals(False)

        # Affiche la première section par défaut si dispo
        if self.sectionsHTML:
            self.section_combo.setCurrentIndex(1)  # <-- ceci déclenchera on_section_changed

    def updateComparison(self):
        titre_choisi = self.combo_titre.currentText()
        indice_choisi_nom = self.combo_indice.currentText()

        if titre_choisi != '' and indice_choisi_nom != '' and self.df_titres_modified is not None:
            
            sectionName = self.compare_section_name

            indice_choisi_ticker = self.liste_indices.get(indice_choisi_nom, "")
            convert_fx = self.checkbox_fx.isChecked()

            # 2. Récupération des données
            first_invest_date = self.df_titres_modified[(self.df_titres_modified["libellé"] == titre_choisi) & (self.df_titres_modified["Opération"].str.contains("Achat|SCRIPT", na=False))]["Date"].min()
            if pd.isna(first_invest_date):
                self.label_status.setText("Achat introuvable pour ce titre.")
            else:
                # Récupère les prix de l'action et de l'indice
                prix_action = self.hist_data[titre_choisi].reindex(self.df_price_total.index).ffill()
                hist_indice = Ticker(indice_choisi_ticker).history(start=self.start_date, end=self.end_date, interval=self.theInterval) #start=first_invest_date, end=df_price_total.index[-1]
                hist_indice.index = hist_indice.index.tz_localize(None)
                prix_indice = hist_indice["Close"].reindex(self.df_price_total.index).ffill()

                if convert_fx and indice_choisi_nom in ["MSCI World", "S&P500", "NASDAQ100"]:
                    # Récupère le taux EUR/USD sur la même période
                    eurusd = Ticker("EURUSD=X").history(start=self.start_date, end=self.end_date, interval=self.theInterval)
                    eurusd.index = eurusd.index.tz_localize(None)
                    taux_eur_usd = eurusd["Close"].reindex(self.df_price_total.index).ffill()
                    # Conversion USD -> EUR (1 USD = X EUR donc 1 EUR = 1/X USD)
                    prix_indice = prix_indice / taux_eur_usd        

                # 3. Calcul base 0% à la date du premier achat
                try:
                    base_action = prix_action.loc[first_invest_date]
                    base_indice = prix_indice.loc[first_invest_date]
                    evol_action = (prix_action / base_action - 1) * 100
                    evol_indice = (prix_indice / base_indice - 1) * 100
                except Exception:
                    self.label_status.setText("Impossible de calculer l'évolution (données manquantes).")
                    evol_action = evol_indice = pd.Series(index=prix_action.index, data=np.nan)

            # 4. Affichage graphique
            fig_compare = Figure()

            # Action : dot avant achat, solide après
            fig_compare.add_trace(Scatter(
                x=evol_action.index,
                y=[evol_action.loc[d] if d <= first_invest_date else None for d in evol_action.index],
                mode='lines',
                name=f"{titre_choisi} (avant achat)",
                line=dict(color='blue', dash='dot'),
                legendgroup=titre_choisi,
                showlegend=False,
                hovertemplate=f"{titre_choisi}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
            ))
            fig_compare.add_trace(Scatter(
                x=evol_action.index,
                y=[evol_action.loc[d] if d >= first_invest_date else None for d in evol_action.index],
                mode='lines',
                name=titre_choisi,
                legendgroup=titre_choisi,
                line=dict(color='blue', dash='solid'),
                hovertemplate=f"{titre_choisi}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
            ))

            # Indice : dot avant achat, solide après
            fig_compare.add_trace(Scatter(
                x=evol_indice.index,
                y=[evol_indice.loc[d] if d <= first_invest_date else None for d in evol_indice.index],
                mode='lines',
                name=f"{indice_choisi_nom} (avant achat)",
                line=dict(color='orange', dash='dot'),
                showlegend=False,
                legendgroup=indice_choisi_nom,
                hovertemplate=f"{indice_choisi_nom}<br>Date: %{{x|%Y-%m-%d}}<br>Évolution: %{{y:.2f}}%<extra></extra>"
            ))
            fig_compare.add_trace(Scatter(
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

            self.addSection(sectionName, fig_compare)

            # Test if sectionName is already in the combo
            already_in_combo = False
            for i in range(self.section_combo.count()):
                if self.section_combo.itemText(i) == sectionName:
                    already_in_combo = True
                    break

            if not already_in_combo:
                idx = self.section_combo.currentIndex()

                self.section_combo.blockSignals(True)
                self.section_combo.clear()
                self.section_combo.addItems(list(self.sectionsHTML.keys()))
                self.section_combo.setCurrentIndex(idx)
                self.section_combo.blockSignals(False)            

            if self.section_combo.itemText(self.section_combo.currentIndex()) == sectionName:
                # force refresh
                idx = self.section_combo.currentIndex()
                self.on_section_changed(idx)       
        else:
            self.sectionsHTML[self.compare_section_name] ='<html><body></body></html>'            

    def on_section_changed(self, idx):
        if idx < 0:
                return        
        key = self.section_combo.itemText(idx)
        if key == self.compare_section_name:
            self.compare_widget.show()
        else:
            self.compare_widget.hide()
        if key and key in self.sectionsHTML:
            content = self.sectionsHTML[key]
            if isinstance(content, QUrl):
                self.plotly_view.load(content)
            else:
                self.plotly_view.setHtml(content)
        self.plotly_view.repaint()

    # def on_section_changed(self, idx):
        
    #     # Récupère le texte de l'item sélectionné
    #     key = self.section_combo.itemText(idx)
    #     if key == self.compare_section_name:
    #         self.compare_widget.show()
    #     else:
    #         self.compare_widget.hide()
    #     if key and key in self.sectionsHTML:
    #         self.plotly_view.setHtml(self.sectionsHTML[key])
    #         self.plotly_view.repaint()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())