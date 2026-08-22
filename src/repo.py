from fpdf import FPDF

# 1. Initialisation de VOTRE rapport existant
pdf = FPDF()
pdf.add_page()

# --- Section précédente (Vos puces par exemple) ---
pdf.set_font("helvetica", "B", 11)
pdf.cell(w=75, h=6, txt="-  Cloisonnement strict (Train/Test) :", ln=0)
pdf.set_font("helvetica", "", 11)
pdf.multi_cell(w=0, h=6, txt="La separation 80% Train / 20% Test a ete effectuee...")

# On ajoute un espace avant le tableau pour souffler
pdf.ln(10)

# --- INTÉGRATION DE VOTRE TABLEAU ICI ---
pdf.set_font("helvetica", "B", 10)
pdf.cell(w=0, h=8, txt="Tableau récapitulatif des performances :", ln=1)
pdf.ln(2)

# Données du tableau
TABLE_DATA = (
    ("Metrique Evaluee", "Regression Logistique", "XGBoost (Optimise)", "Evolution / Impact"),
    ("AUC-ROC", "0.8413", "0.8463", "+0.0050 (Legere hausse)"),
    ("Accuracy globale", "73.81 %", "80.48 %", "+6.67 % (Forte progression)"),
    ("Precision (Classe 1)", "0.50", "0.68", "+0.18 (Moins de fausses alertes)"),
    ("Rappel / Recall", "0.78", "0.50", "-0.28 (Plus de clients rates)"),
    ("F1-Score (Classe 1)", "0.61", "0.58", "-0.03 (Legere baisse globale)")
)

pdf.set_font("helvetica", size=9) # Taille de police adaptée pour le tableau

# Génération automatique du tableau dans le rapport
with pdf.table(borders_layout="GRID", col_widths=(50, 40, 40, 60), cell_fill_color=245, cell_fill_mode="ROWS") as table:
    for data_row in TABLE_DATA:
        row = table.row()
        for datum in data_row:
            row.cell(datum)

# --- Suite du rapport ---
pdf.ln(10)
pdf.set_font("helvetica", "", 11)
pdf.multi_cell(w=0, h=6, txt="En conclusion, le modele XGBoost apporte une nette amelioration...")

# 2. SAUVEGARDE UNIQUE DE VOTRE RAPPORT
pdf.output("./rapport.pdf")
