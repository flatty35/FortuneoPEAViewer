import sys
import io
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QDateEdit, QCheckBox, QDockWidget, QLineEdit, QToolButton, QGridLayout, QSizePolicy,
    QScrollArea, QHeaderView
)
from PySide6.QtCore import Qt, QDate, QDir, QUrl, QSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objs as go
import plotly.io as pio

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
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.tickersRecallLayout = QHBoxLayout()
        self.btn_show_dock = QPushButton("Tickers")
        self.btn_show_dock.setVisible(False)
        self.tickersRecallLayout.addStretch()
        self.tickersRecallLayout.addWidget(self.btn_show_dock)
        left_layout.addLayout(self.tickersRecallLayout)
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

        left_layout.addWidget(import_toggle)
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

        # Checkbox for FX
        self.checkbox_fx = QCheckBox("Intégrer le taux de change EUR/USD pour les indices internationaux")
        self.checkbox_fx.setChecked(True)
        options_layout.addWidget(self.checkbox_fx)

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

        # Création du widget contenant le QWebEngineView
        plotly_container = QWidget()
        plotly_layout = QVBoxLayout(plotly_container)
        self.plotly_view = QWebEngineView()
        self.plotly_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plotly_layout.addWidget(self.plotly_view)

        # Création de la scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(plotly_container)

        # Ajout au layout principal
        left_layout.addWidget(scroll_area, stretch=2)

        # Status label
        self.status_group = QGroupBox("Status") #Import des fichiers
        self.status_layout = QVBoxLayout()
        self.label_status = QLabel("Veuillez charger les fichiers.")
        self.status_layout.addWidget(self.label_status)
        self.status_group.setLayout(self.status_layout)
        left_layout.addWidget(self.status_group)

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
        self.checkbox_fx.stateChanged.connect(self.process_and_plot)
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
                self.process_and_plot()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de lecture du fichier titres : {e}")

    def load_especes(self):
        path = self.file_fields['especes_file'][0].text()
        if path:
            try:
                self.df_especes = pd.read_csv(path, encoding="latin1", sep=";")
                self.label_status.setText("Historique espèces chargé.")
                self.process_and_plot()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur de lecture du fichier espèces : {e}")

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

    def process_and_plot(self):
        # Vérifie que les deux fichiers sont chargés
        if self.df_titres is None or self.df_especes is None:
            self.label_status.setText("Veuillez charger les deux fichiers.")
            return

        figs = []
        try:
            # Nettoyage et parsing des dates
            df_titres = self.df_titres.copy()
            df_especes = self.df_especes.copy()
            df_titres["Date"] = pd.to_datetime(df_titres["Date"], dayfirst=True, errors='coerce')
            df_especes["Date opération"] = pd.to_datetime(df_especes["Date opération"], dayfirst=True, errors='coerce')
            df_especes["Date valeur"] = pd.to_datetime(df_especes["Date valeur"], dayfirst=True, errors='coerce')

            # Détermine la période sélectionnée
            all_dates = pd.concat([
                df_titres["Date"], 
                df_especes["Date opération"], 
                df_especes["Date valeur"]
            ]).dropna()
            if all_dates.empty:
                self.label_status.setText("Aucune date valide trouvée dans les fichiers.")
                return

            start = all_dates.min()
            end = all_dates.max()

            # self.date_start.setDate(QDate(min_date.year, min_date.month, min_date.day))
            # self.date_end.setDate(QDate(max_date.year, max_date.month, max_date.day))

            # # Met à jour les sélecteurs de date si besoin
            # # if not self.date_start.date().isValid() or not self.date_end.date().isValid():
            # start = self.date_start.date().toPython()
            # end = self.date_end.date().toPython()

            # Filtre les données selon la période sélectionnée
            df_titres = df_titres[(df_titres["Date"] >= start) & (df_titres["Date"] <= end)]
            df_especes = df_especes[(df_especes["Date opération"] >= start) & (df_especes["Date opération"] <= end)]

            # Traitement des montants
            if "Montant net" in df_titres.columns:
                df_titres["Montant net"] = pd.to_numeric(df_titres["Montant net"].astype(str).str.replace(",", "."), errors='coerce').fillna(0)
            if "Crédit" in df_especes.columns:
                df_especes["Crédit"] = pd.to_numeric(df_especes["Crédit"].astype(str).str.replace(",", "."), errors='coerce').fillna(0)

            # Cumul des versements espèces
            df_versements = df_especes[df_especes["libellé"].str.lower().str.contains("versement", na=False)]
            df_versements = df_versements[["Date opération", "Crédit"]].dropna()
            df_versements = df_versements.rename(columns={"Date opération": "Date", "Crédit": "Montant net"})
            df_montantsNet = df_titres[["Date", "Montant net"]].dropna()
            df_allOperations = pd.concat([df_versements, df_montantsNet], axis=0, ignore_index=True)
            df_allOperations = df_allOperations.groupby("Date")["Montant net"].sum()
            df_compte_espece = df_allOperations.cumsum()
            df_versements_cum = df_versements.groupby("Date")["Montant net"].sum().cumsum()

            # Prépare le graphique Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_compte_espece.index, y=df_compte_espece.values,
                mode='lines+markers', name="Compte Espèce"
            ))
            fig.add_trace(go.Scatter(
                x=df_versements_cum.index, y=df_versements_cum.values,
                mode='lines+markers', name="Versements cumulés"
            ))
            fig.update_layout(
                title="Compte Espèce et Versements cumulés",
                xaxis_title="Date",
                yaxis_title="Montant (€)",
                legend_title="Légende",
                template="plotly_white"
            )
            figs.append(fig)
            
            self.label_status.setText("Graphique mis à jour.")
        except Exception as e:
            self.label_status.setText(f"Erreur lors du traitement : {e}")


        # insérer la logique  de FortuneoPEAViewer.py ici

        self.show_plotly([fig])            

        # --- Place here all the logic from FortuneoPEAViewer.py ---
        # 1. Parse dates, clean data, compute all DataFrames as in your Streamlit code
        # 2. Use self.combo_interval.currentText(), self.date_start.date(), self.date_end.date(), self.checkbox_fx.isChecked()
        # 3. Build your plotly figures (go.Figure)
        # 4. Render with: self.show_plotly(fig)

        # # Example: show a dummy plotly chart
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=[1,2,3], y=[1,4,9], mode='lines', name='Exemple'))
        # fig.update_layout(title="Exemple Plotly dans PySide6")
        # self.show_plotly([fig, fig, fig])

    def show_plotly(self, figs):
        # Convert the Plotly figure to HTML and display it in the QWebEngineView
        html = ""
        # Concatène les HTML de chaque figure
        html = ""
        for i, fig in enumerate(figs):
            # Ajoute un séparateur visuel entre les graphes si plusieurs
            if i > 0:
                html += "<hr style='margin:40px 0;'>"
            html +=  pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            
        self.plotly_view.setHtml(html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())