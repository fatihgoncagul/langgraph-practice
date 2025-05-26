import psycopg2
from langchain_core.tools import tool

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "healthcare",
    "user": "postgres",
    "password": "753258"
}

@tool
def get_patient_vitals(patient_id: int) -> str:
    """Verilen patient_id göre veritabanından hastanın vitallerini yani vital bilgilerini getirir."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT age, gender, blood_type, height_cm, weight_kg, bmi,
                   temperature_c, heart_rate_bpm, blood_pressure_mmhg
            FROM patients
            WHERE patient_id = %s
        """, (patient_id,))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if row is None:
            return "Bu ID'ye sahip bir hasta bulunamadı."

        return (
            f"Hasta {patient_id}: "
            f"{row[1]} cinsiyetinde, {row[0]} yaşında. "
            f"Kan grubu {row[2]}, Boy: {row[3]:.2f} cm, Kilo: {row[4]:.2f} kg, "
            f"BMI: {row[5]:.2f}, Ateş: {row[6]:.2f}°C, Nabız: {row[7]} bpm, Tansiyon: {row[8]}."
        )

    except Exception as e:
        return f"Hasta bilgisi alınırken hata oluştu: {str(e)}"

@tool
def get_lab_results(patient_id: int) -> str:
    """Verilen patient_id göre birden fazla laboratuvar sonucunu listeler."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT result_date, hemoglobin, hematocrit, wbc, rbc, platelets,
                   mch, mchc, mcv, rdw, mpv, notes
            FROM lab_results
            WHERE patient_id = %s
            ORDER BY result_date DESC
        """, (patient_id,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return f"Hasta {patient_id} için laboratuvar sonucu bulunamadı."

        result_list = []
        for row in rows:
            result_list.append(
                f"📅 Tarih: {row[0]}\n"
                f"- Hemoglobin: {row[1]} g/dL\n"
                f"- Hematokrit: {row[2]} %\n"
                f"- WBC: {row[3]} 10⁹/L\n"
                f"- RBC: {row[4]} 10¹²/L\n"
                f"- Platelet: {row[5]} /µL\n"
                f"- MCH: {row[6]} pg\n"
                f"- MCHC: {row[7]} g/dL\n"
                f"- MCV: {row[8]} fL\n"
                f"- RDW: {row[9]} %\n"
                f"- MPV: {row[10]} fL\n"
               # f"Not: {row[11]}\n"
               
            )

        return f"Hasta {patient_id} için laboratuvar geçmişi:\n\n" + "\n\n".join(result_list)

    except Exception as e:
        return f"Laboratuvar sonuçları alınırken hata oluştu: {str(e)}"
