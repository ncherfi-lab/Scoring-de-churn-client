# src/generate_report.py
from fpdf import FPDF
import os

class ChurnReportPDF(FPDF):
    def header(self):
        #  CORRECTIF 1 : L'en-tête ne s'affiche JAMAIS sur la page de garde (page 1)
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(140, 140, 140)
            self.cell(0, 10, "TelcoWave - Strategie Operationnelle Anti-Churn", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(10, 18, 200, 18) # Ligne plus fine et mieux positionnée
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def check_page_break(self, space_needed=40):
        #  CORRECTIF 2 : Si l'espace restant sur la page est trop faible, on saute à la page suivante
        if self.get_y() > (self.page_break_trigger - space_needed):
            self.add_page()

    def add_chapter_title(self, label):
        self.check_page_break(45) # Évite un titre isolé en bas de page
        self.ln(6)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(24, 43, 73) # Bleu marine professionnel
        self.cell(0, 10, label, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_sub_chapter_title(self, label):
        self.check_page_break(30)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def add_body_text(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def add_bullet_point(self, bold_prefix, text):
        self.check_page_break(20)
        self.set_font("Helvetica", "B", 10.5)
        self.cell(8, 6, "  . ", align="L")
        self.cell(self.get_string_width(bold_prefix) + 1, 6, bold_prefix, align="L")
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 6, text)
        self.ln(1.5)

def build_pdf_report(output_path="reports/rapport_projet_Churn.pdf"):
    print("🚀 Generation du rapport graphique optimise...")
    
    pdf = ChurnReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20) # Marge basse augmentee pour respirer
    pdf.add_page()
    
    # =========================================================================
    # PAGE 1 : PAGE DE GARDE NETTE & PROPRE
    # =========================================================================
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(24, 43, 73)
    pdf.multi_cell(0, 12, "RAPPORT DE SYNTHESE FINAL\nOptimisation & Evaluation du Modele de Churn", align="C")
    
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, "Projet Industriel - Retention Clientele TelcoWave", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Livrable Officiel - Statut : Valide et Gele", align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Encadré visuel pour les métadonnées (Auteur / Date)
    pdf.ln(50)
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(220, 225, 235)
    pdf.set_x(40)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(50, 50, 50)
    
    meta_text = (
        "  Auteur : Data Scientist Consultant\n"
        "  Date de livraison : 04 Septembre 2026\n"
        "  Format : Livrable Operationnel de Fin de Formation"
    )
    pdf.multi_cell(130, 7, meta_text, border=1, fill=True)
    
    # =========================================================================
    # PAGE 2 : INTRODUCTION ET FEATURE ENGINEERING
    # =========================================================================
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    
    pdf.add_body_text("Ce rapport presente l'integralite des resultats, des analyses d'interpretabilite "
                      "et des arbitrages strategiques issus de l'etude predictive de l'attrition client (churn) "
                      "menee sur les donnees de l'operateur TelcoWave.")
    
    pdf.add_chapter_title("1. Demarche et Feature Engineering")
    pdf.add_body_text("Pour capturer les signaux comportementaux des clients, le jeu de donnees initial a ete "
                      "enrichi par la creation de variables specifiques. Afin de garantir la robustesse globale "
                      "de cette approche, un protocole rigoureux a ete mene :")
    
    pdf.add_bullet_point("Cloisonnement strict (Train/Test) : ", 
                         "La separation 80% Train / 20% Test a ete effectuee a la racine pour sanctuariser "
                         "le jeu de test independant et eliminer tout risque de Data Leakage (fuite de donnees).")
    
    pdf.add_bullet_point("Standardisation des donnees: ", 
                         "Nous avons integre un ColumnTransformer appliquant un OneHotEncoder sur les variables "
                         "categorielles et un StandardScaler sur les variables numeriques. Les echelles de nos variables "
                         "etant deja bien proportionnees, un simple StandardScaler suffit a harmoniser les echelles.")
    
    pdf.add_bullet_point("Selection par RFECV : ", 
                         "Sur les 54 variables generees apres encodage, l'algorithme a automatiquement rejete "
                         "16 colonnes redondantes, verrouillant le modele sur un coeur de 38 variables hautement predictives.")
    # --- PREMIÈRE PUCE ---
    # On fixe la colonne de gauche à une largeur de 75 mm (ajustez selon vos besoins)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(w=75, h=6, txt="- Cloisonnement strict (Train/Test) :", ln=0)
    # La description prend tout le reste de la ligne (w=0) et revient à la ligne à la fin
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(w=0, h=6, txt="La separation 80% Train / 20% Test a ete effectuee a la racine pour sanctuariser le jeu de test independant et eliminer tout risque de Data Leakage (fuite de donnees).")

    # Petit espace vertical entre les deux blocs de puces
    pdf.ln(4)

    # --- DEUXIÈME PUCE ---
    # On utilise EXACTEMENT la même largeur (w=75) pour que le texte s'aligne au même endroit
    pdf.cell(w=75, h=6, txt=".  Standardisation des données:", ln=0)
    pdf.multi_cell(w=0, h=6, txt="Nous avons integre un ColumnTransformer appliquant un OneHotEncoder...")

    # =========================================================================
    # PAGE 3 : RÉSULTATS & CALIBRATION (FUSIONNÉS PROPREMENT)
    # =========================================================================
    pdf.add_chapter_title("2. Match des Champions : Resultats sur le Jeu de Test")
    pdf.add_body_text("Une recherche d'hyperparametres intensive (RandomizedSearchCV, duree : 30min 53sec) a permis "
                      "de stabiliser le modele XGBoost. Confronte a la Regression Logistique (Baseline), "
                      "les scores d'AUC confirment une excellente robustesse globale :")
    
    pdf.add_bullet_point("Regression Logistique (Baseline) : ", "AUC = 0.8413")
    pdf.add_bullet_point("XGBoost Optimise (Final) : ", "AUC = 0.8463")
    
    pdf.add_body_text("L'ecart technique de seulement 0.005 (0.5 %) place les deux modeles au meme niveau d'excellence. "
                      "Cependant, l'analyse montre que la recherche a bride la puissance de XGBoost pour eliminer le surapprentissage, "
                      "offrant un gain de +2.8 points sur le jeu de test (max_depth=3, learning_rate=0.05).")

    pdf.add_chapter_title("3. Diagnostic de la Calibration et Brier Score")
    pdf.add_body_text("Pour valider l'utilisation commerciale directe des probabilites calculees, la calibration a ete "
                      "mesuree sur l'echantillon de test independant :")
    pdf.add_bullet_point("Brier Score Baseline : ", "0.1688")
    pdf.add_bullet_point("Brier Score XGBoost : ", "0.1354 (Plus proche de 0, donc significativement plus precis).")
    
    pdf.add_body_text("L'analyse visuelle confirme que la courbe de calibration de XGBoost est tres proche de la diagonale de reference. "
                      "Ses probabilites de risque sont mathematiquement fiables. Aucun recalibrage n'est necessaire.")

    # =========================================================================
    # PAGE 4 : LES GRAPHIQUES (CENTRÉS ET ASSOCIES)
    # =========================================================================
    pdf.add_page()
    pdf.add_chapter_title("4. Visualisations des Performances Graphiques")
    
    # 🚨 CORRECTIF 3 : On affiche l'image de la courbe ROC de taille raisonnable et centree
    fig_path1 = "reports/figures/Courbe ROC - XGBoost.png"
    if os.path.exists(fig_path1):
        pdf.image(fig_path1, x=55, w=100) # w=100 au lieu de 140 pour ne pas saturer l'espace
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.cell(0, 6, "Figure 4.1 : Courbe ROC comparative - Performance de classement", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # =========================================================================
    # PAGE 5 : OPTIMISATION FINANCIÈRE
    # =========================================================================
    pdf.add_chapter_title("5. Optimisation Financiere du Seuil de Decision")
    
    # Mise en valeur de la contrainte budgetaire dans un encadre gris
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, " Contrainte Economique de la campagne TelcoWave :\n"
                         " - Cout d'envoi de l'offre (Faux Positif) = 15 EUR\n"
                         " - Valeur client sauvee (Vrai Positif) = 120 EUR\n"
                         " -> Ratio d'asymetrie financiere fort de 1 a 8.", border=1, fill=True)
    pdf.ln(4)

    pdf.add_bullet_point("Strategie A (Approche ROI) : ", 
                         "L'optimisation du profit fixe le seuil de declenchement a 12 % de probabilite de churn. "
                         "A ce niveau, le profit net maximal genere atteint 29.56 kEUR sur l'echantillon.")
    
    pdf.add_bullet_point("Strategie B (Approche Budgetaire) : ", 
                         "Le tri des clients par risque decroissant montre que pour capter ce profit maximal, "
                         "l'entreprise doit cibler exactement le Top 60 % des clients les plus instables.")
    
    pdf.add_body_text("Le Lien Logique : Le Top 60 % des clients les plus dangereux correspond tres exactement a la population "
                      "affichant une probabilite de churn superieure ou egale a 12 % (proba >= 0.12). Les deux strategies ")

    pdf.output(output_path)
        
    print(f"✅ Succès ! Le fichier PDF a été créé ici : {output_path}")
# --- LE BLOC INDISPENSABLE POUR COMPILER ---
if __name__ == "__main__":
    build_pdf_report()