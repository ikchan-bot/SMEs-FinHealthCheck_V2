import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import plotly.graph_objects as go
import streamlit.components.v1 as components
from autogluon.tabular import TabularPredictor

# ==========================================
# 1. ตั้งค่าหน้าเว็บและธีม (NOMOS Style)
# ==========================================
st.set_page_config(
    page_title="SME FinCheck",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;600&family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], p, div, label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, h1, h2, h3, h4, h5, h6, .stTitle, button, input, select, a {
        font-family: 'Sarabun', sans-serif !important;
    }
    
    h1, h2, h3 { color: #1E3A8A !important; font-weight: 600; }
    
    /* ตกแต่งปุ่ม Primary และ Link Button (สีชมพูจุฬาฯ) */
    div[data-testid="stBaseButton-primary"] > button, button[kind="primary"], div[data-testid="stLinkButton"] > a {
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        border: 2px solid #A9A9A9 !important;
        background-color: white !important;
        color: #333 !important;
        text-decoration: none !important;
    }
    div[data-testid="stBaseButton-primary"] > button:hover, button[kind="primary"]:hover, div[data-testid="stLinkButton"] > a:hover {
        background-color: #FE5C8D !important;
        border-color: #FE5C8D !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(254, 92, 141, 0.4) !important;
        transform: scale(1.02) !important;
    }
    
    .hero-title {
        font-family: 'Sarabun', sans-serif !important;
        font-size: 2.5em !important;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        line-height: 1.3;
    }
    .hero-subtitle {
        font-family: 'Sarabun', sans-serif !important;
        font-size: 1.2em !important;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ระบบจัดการ Session State
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

def scroll_to_top():
    js = """
        <script>
            setTimeout(function() {
                window.parent.scrollTo(0, 0);
                var containers = window.parent.document.querySelectorAll('.main, .block-container, .stApp');
                for (var i = 0; i < containers.length; i++) { containers[i].scrollTop = 0; }
            }, 150);
        </script>
    """
    components.html(js, height=0)

# ==========================================
# 3. ฟังก์ชันโหลดโมเดล (สมองใหม่ V2)
# ==========================================
def assemble_model_parts():
    output_path = "models/Y_BFC_Model_Full.zip" 
    if os.path.exists(output_path):
        return output_path
        
    part_files = [
        "models/Model.part1.zip", 
        "models/Model.part2.zip",
        "models/Model.part3.zip",
        "models/Model.part4.zip",
        "models/Model.part5.zip"
    ]
    
    with open(output_path, 'wb') as outfile:
        for part in part_files:
            if os.path.exists(part):
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
            else:
                st.error(f"❌ ไม่พบไฟล์ {part} กรุณาตรวจสอบในโฟลเดอร์ models/")
                return None
    return output_path

@st.cache_resource
def load_resources():
    # 1. โหลด AutoGluon
    model_zip_path = assemble_model_parts()
    extract_path = './models/autogluon_extracted'
    
    if model_zip_path and not os.path.exists(extract_path):
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
    model_path = extract_path
    if os.path.exists(extract_path):
        for root, dirs, files in os.walk(extract_path):
            if 'predictor.pkl' in files:
                model_path = root
                break
                
    try:
        predictor = TabularPredictor.load(model_path, require_py_version_match=False)
    except:
        predictor = None

    # 2. โหลด KMeans
    try:
        kmeans = joblib.load('models/KMeans_model_7Clusters.pkl')
        scaler = None 
    except:
        kmeans = None
        scaler = None

    # 3. โหลด Mock Data
    try:
        df_raw = pd.read_excel('data/mock_data.xlsx')
    except:
        df_raw = pd.DataFrame()

    return kmeans, scaler, predictor, df_raw

kmeans_model, scaler_model, predictor_model, df_raw = load_resources()

# ==========================================
# 4. หน้า Landing Page
# ==========================================
def show_landing():
    c_img1, c_img2, c_img3 = st.columns([1, 2, 1]) 
    with c_img2:
        try:
            st.image("FinCheck.jpg", use_container_width=True) 
        except:
            pass # ถ้าไม่มีรูปก็ข้ามไป ไม่ให้ระบบพัง

    st.markdown("""
        <div class="hero-title">
            ตรวจสุขภาพธุรกิจและการเงินด้วย<br>
            <span style='font-family: "Jost", sans-serif; font-weight: 500; color: #FE5C8D; font-size: 1.1em;'>SME FinCheck v2</span>
        </div>
        <div class="hero-subtitle">รู้ทันสุขภาพการเงิน | ประเมิน DNA ธุรกิจ | ลดความเสี่ยง | รับคำแนะนำ</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1]) 
    with c_btn2:
        if st.button("🚀 เริ่มประเมินทันที (Start)", type="primary", use_container_width=True):
            navigate_to('input_step1')
    st.markdown("---")

    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;'>
        พัฒนาโดย: <b>นายสมเกียรติ จูสวัสดิ์</b><br>
        นิสิตปริญญาเอก | หลักสูตรธุรกิจเทคโนโลยีและการจัดการนวัตกรรม<br>
        จุฬาลงกรณ์มหาวิทยาลัย
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. หน้า Input Step 1
# ==========================================
def show_input_step1():
    scroll_to_top()
    st.markdown('<p style="color: #888; font-size: 1.1em; margin-bottom: 0;">ขั้นตอนที่ 1/2: การประเมิน</p>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1E3A8A;'>🧬 DNA ธุรกิจของท่าน</h3>", unsafe_allow_html=True)
    st.info("💡 **กรุณาประเมินระดับการดำเนินงาน**\n\n**0** = ไม่มี &nbsp;&nbsp;•&nbsp;&nbsp; **1** = น้อยที่สุด &nbsp;&nbsp;•&nbsp;&nbsp; **5** = มากที่สุด")

    score_options = [0, 1, 2, 3, 4, 5]

    with st.form("form_step1"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>การตลาดและผลิตภัณฑ์</h5>", unsafe_allow_html=True)
            beh_mon = st.selectbox("ท่านติดตามและตรวจสอบความพึงพอใจของลูกค้า", score_options, index=0)
            brn_image = st.selectbox("ท่านให้ความสำคัญกับภาพลักษณ์องค์กร", score_options, index=0)
            brn_brand = st.selectbox("การรับรู้และความน่าเชื่อถือของแบรนด์ของท่าน", score_options, index=0)
        
        with col2:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>เทคโนโลยีและการรับมือสถานการณ์</h5>", unsafe_allow_html=True)
            sav_virus = st.selectbox("การอัพเดทโปรแกรมป้องกันไวรัสเพื่อความปลอดภัย", score_options, index=0)
            sav_pdpa = st.selectbox("ท่านปฏิบัติตามกฎหมาย PDPA", score_options, index=0)
            cri_pln = st.selectbox("ท่านมีแผนรองรับวิกฤตการณ์ต่าง ๆ", score_options, index=0)

        with col3:
            st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>นโยบายภาครัฐ</h5>", unsafe_allow_html=True)
            pol_ben = st.selectbox("ท่านได้รับประโยชน์จากนโยบายภาครัฐ", score_options, index=0)
            pol_adj = st.selectbox("ปรับรูปแบบธุรกิจให้สอดคล้องนโยบายรัฐ", score_options, index=0)

        st.markdown("---")
        if st.form_submit_button("ถัดไป >", type="primary", use_container_width=True):
            st.session_state.inputs.update({
                'BEH_MON': beh_mon, 'BRN_IMAGE': brn_image, 'BRN_BRAND': brn_brand,
                'SAV_VIRUS': sav_virus, 'SAV_PDPA': sav_pdpa, 'CRI_PLN': cri_pln,
                'POL_BEN': pol_ben, 'POL_ADJ': pol_adj
            })
            navigate_to('input_step2')

# ==========================================
# 6. หน้า Input Step 2
# ==========================================
def show_input_step2():
    scroll_to_top() 
    st.markdown('<p style="color: #888; font-size: 1.1em; margin-bottom: 0;">ขั้นตอนที่ 2/2: การประเมิน</p>', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1E3A8A; margin-top: 0;'>💼 ระดับดำเนินงานและงบการเงิน</h3>", unsafe_allow_html=True)
    st.info("💡 **กรุณาประเมินระดับการดำเนินงาน และกรอกข้อมูลทางการเงินเพื่อประมวลผล**\n\n**0** = ไม่มี   •   **1** = น้อยที่สุด   •   **5** = มากที่สุด")

    score_options = list(range(6))
    binary_options = ["ไม่มี (0)", "มี (1)"]

    with st.form("form_step2"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>ผู้ประกอบการและทีมงาน</p>", unsafe_allow_html=True)
            cap_netw = st.selectbox("ท่านใช้เครือข่ายหรือพันธมิตรในการดำเนินธุรกิจ", score_options, index=0)
            csr3 = st.selectbox("กิจการของท่านมีระบบกำจัดของเสีย", binary_options, index=0)
            ohr_org = st.selectbox("กิจการของท่านมีการจัดผังโครงสร้างองค์กร", binary_options, index=0)
        
        with col2:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>การบัญชีและสถานการณ์เศรษฐกิจ</p>", unsafe_allow_html=True)
            prc_cfw = st.selectbox("กระแสเงินสดเพื่อประกอบธุรกิจและชำระหนี้", score_options, index=0)
            eco_adt = st.selectbox("ความสามารถในการปรับตัวรับสถานการณ์เศรษฐกิจ", score_options, index=0)
        
        with col3:
            st.markdown("<p style='color: #1E3A8A; font-weight: bold;'>เทคโนโลยีและการสื่อสาร</p>", unsafe_allow_html=True)
            ecm_net = st.selectbox("การเข้าถึงเครือข่ายอินเตอร์เน็ตของกิจการ", score_options, index=0)
            res_ch = st.selectbox("ความสามารถในการโต้ตอบลูกค้า", score_options, index=0)

        st.markdown("---")
        
        st.markdown("<h5 style='color: #1E3A8A; font-weight: bold;'>📊 ข้อมูลทางการเงิน (หน่วย: บาท)</h5>", unsafe_allow_html=True)
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            avg3y_ni_raw = st.number_input("1. กำไรสุทธิเฉลี่ย 3 ปี (บาท)", value=0.0, step=1000.0, format="%.2f")
        with col_fin2:
            avg3y_rev_raw = st.number_input("2. ยอดขายรวมเฉลี่ย 3 ปี (บาท)", min_value=1.0, value=100000.0, step=1000.0, format="%.2f")

        st.markdown("---")
        
        if st.form_submit_button("🚀 ประเมินผลลัพธ์", type="primary", use_container_width=True):
            
            # การคำนวณอัตราส่วนและ Interaction Variables
            avg3y_ni_ratio = (avg3y_ni_raw * 100) / avg3y_rev_raw
            sav_pdpa = st.session_state.inputs.get('SAV_PDPA', 0)
            
            sav_pdpa_x_prc_cfw = sav_pdpa * prc_cfw
            prc_cfw_x_avg3y_ni = prc_cfw * avg3y_ni_ratio
            
            # อัปเดตข้อมูลทั้งหมดลง Session State
            st.session_state.inputs.update({
                'CAP_NETW': cap_netw, 
                'CSR3': 1 if "มี" in csr3 else 0, 
                'OHR_ORG': 1 if "มี" in ohr_org else 0,
                'PRC_CFW': prc_cfw, 
                'ECO_ADT': eco_adt,
                'ECM_NET': ecm_net, 
                'RES_CH': res_ch,
                'AVG_3Y_NI': avg3y_ni_raw,
                'AVG_3Y_REV': avg3y_rev_raw,
                'Avg3Y_NI_Ratio': avg3y_ni_ratio,
                'SAV_PDPA_x_PRC_CFW': sav_pdpa_x_prc_cfw,
                'PRC_CFW_x_Avg3Y_NI': prc_cfw_x_avg3y_ni
            })
            
            success = process_results()
            if success:
                navigate_to('dashboard')

# ==========================================
# 7. ฟังก์ชันประมวลผล (Processing Logic)
# ==========================================
def process_results():
    prob = 0.5
    cluster_id = 0
    inputs = st.session_state.inputs

    # 1. Clustering Logic
    cluster_features = ['BEH_MON', 'BRN_IMAGE', 'BRN_BRAND', 'SAV_VIRUS', 'SAV_PDPA', 'CRI_PLN', 'POL_BEN', 'POL_ADJ']
    cluster_vals = [inputs.get(f, 0) for f in cluster_features]
    
    try:
        X_cluster = pd.DataFrame([cluster_vals], columns=cluster_features)
        if scaler_model is not None:
            X_cluster = scaler_model.transform(X_cluster)
        
        raw_cluster_id = kmeans_model.predict(X_cluster)
        cluster_id = int(np.ravel(raw_cluster_id).item())
    except Exception as e:
        print(f"Cluster Error: {e}")
        cluster_id = 0
        
    st.session_state.results['cluster_id'] = cluster_id

    # 2. Prediction Logic
    if predictor_model is not None and not df_raw.empty:
        try:
            pred_df = df_raw.head(1).copy().reset_index(drop=True)
            
            for col in pred_df.columns:
                pred_df.at[0, col] = float('nan')
            
            # โยนตัวแปรทั้งหมดจากหน้าเว็บลง DataFrame (ต้องแน่ใจว่าชื่อ Key ตรงกับชื่อ Column ใน Excel)
            for key, val in inputs.items():
                if key in pred_df.columns:
                    pred_df.at[0, key] = float(val) if isinstance(val, (int, float)) else val

            # ข้อมูลบังคับ (Default)
            if 'SIZ' in pred_df.columns: pred_df.at[0, 'SIZ'] = 1 
            if 'YER' in pred_df.columns: pred_df.at[0, 'YER'] = 10 
                
            prob_df = predictor_model.predict_proba(pred_df)
            prob_array = prob_df.values.flatten()
            prob = float(prob_array[-1]) 
            
            # การแปลงสเกล
            MAX_RAW_PROB = 0.491  
            MIN_RAW_PROB = 0.099  
            
            if MAX_RAW_PROB > MIN_RAW_PROB:
                scaled_prob = (prob - MIN_RAW_PROB) / (MAX_RAW_PROB - MIN_RAW_PROB)
            else:
                scaled_prob = 0
                
            risk_score = scaled_prob * 100
            risk_score = min(100.0, max(0.0, risk_score))

        except Exception as e:
            st.error(f"🚨 ข้อผิดพลาดจากระบบพยากรณ์: {e}")
            return False 
    else:
        score = inputs.get('PRC_CFW', 0) * 0.4 + inputs.get('CAP_NETW', 0) * 0.3 + inputs.get('BEH_MON', 0) * 0.3
        prob = 1 - (score / 5.0)
        risk_score = prob * 100

    st.session_state.results['risk_prob'] = prob          
    st.session_state.results['risk_score'] = risk_score   
    
    return True

# ==========================================
# 8. หน้า Dashboard
# ==========================================
def show_dashboard():
    scroll_to_top() 
    
    if 'inputs' not in st.session_state or not st.session_state.inputs:
        st.warning("⚠️ กรุณากรอกข้อมูลในขั้นตอนที่ 1 และ 2 ให้ครบถ้วนก่อนครับ")
        if st.button("กลับไปกรอกข้อมูล"):
            navigate_to('input_step1')
        return

    cluster_id = st.session_state.results.get('cluster_id', 1)
    risk_score = st.session_state.results.get('risk_score', 50.0)
    
    if isinstance(cluster_id, (np.ndarray, list)):
        cluster_id = int(cluster_id)
    
    cluster_info = {
        0: {"name": "Active Marketer (นักการตลาดไฟแรง)", "color": "#F9D607", 
            "desc": "โดดเด่นด้านการตลาดและภาพลักษณ์องค์กร ควรเสริมสร้างระบบเทคโนโลยีและการบริหารความเสี่ยงหลังบ้าน"},
        1: {"name": "Potential Starter (นักสู้ผู้มีศักยภาพ)", "color": "#e74c3c", 
            "desc": "มีความยืดหยุ่น ควรสร้างวินัยทางการเงินและวางระบบบัญชีให้น่าเชื่อถือ เพื่อเพิ่มโอกาสเข้าถึงแหล่งเงินทุน"},
        2: {"name": "Master Leader (ผู้นำระดับมาสเตอร์)", "color": "#2ecc71", 
            "desc": "ความพร้อมรอบด้าน ทั้งด้านการเงิน การตลาด และการรับมือวิกฤตการณ์ ธนาคารและนักลงทุนพร้อมสนับสนุนแหล่งเงินทุน"}
    }
    
    dna = cluster_info.get(cluster_id, cluster_info[1])

    st.markdown(f"<h3 style='text-align:center; color:#1E3A8A;'>📊 ผลการประเมินสุขภาพการเงิน</h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧬 DNA ธุรกิจของคุณ", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: {dna['color']}; padding: 20px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style='margin:0; font-family: Sarabun, sans-serif; color: white !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>{dna['name']}</h3>
            <p style='margin-top:10px; font-size: 1.1em; font-family: Sarabun, sans-serif;'>{dna['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 💡 คำแนะนำเบื้องต้น:", unsafe_allow_html=True)
        
        if cluster_id == 1:
            st.warning("⚠️ ควรเร่งจัดทำบัญชีรายรับ-รายจ่ายให้ชัดเจน และลดภาระหนี้ที่ไม่จำเป็น")
        elif cluster_id == 0:
            st.info("ℹ️ การตลาดยอดเยี่ยม เข้าใจผู้บริโภค แต่ต้องอุดรูรั่วความปลอดภัยของระบบ IT")
        else:
            st.success("✅ เครดิตดี เตรียมเอกสารยื่นกู้เพื่อขยายกิจการได้เลย")

    with col2:
        st.markdown(f"### 🔮 มีข้อจำกัดการเข้าถึงแหล่งเงินทุน: **{risk_score:.1f}%**", unsafe_allow_html=True)
        
        if risk_score < 40:
            risk_level_text = "ต่ำ"
            text_color = "#1b5e20" 
        elif risk_score <= 70:
            risk_level_text = "ปานกลาง"
            text_color = "#b8860b" 
        else:
            risk_level_text = "สูง"
            text_color = "#842029" 

        fig = go.Figure(go.Indicator(
            mode = "gauge",  
            value = risk_score,
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray", 'tickvals': [0, 40, 70, 100]},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},   
                    {'range': [40, 70], 'color': "#F9D607"},  
                    {'range': [70, 100], 'color': "#e74c3c"}  
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        fig.add_annotation(
            x=0.5, y=0.10,  
            text=f"<b>{risk_level_text}</b>",
            font=dict(size=60, color=text_color, family="Sarabun"),
            showarrow=False
        )
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), font={'family': "Sarabun"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    c_btn1, c_btn2, c_btn3 = st.columns([0.15, 0.7, 0.15])
    with c_btn2:
        if st.button("📄 ดูข้อเสนอแนะโดยละเอียด (Recommendation)", type="primary", use_container_width=True):
            navigate_to('recommendation')

# ==========================================
# 9. หน้า Recommendation
# ==========================================
def show_recommendation():
    scroll_to_top() 

    st.markdown("<h3 style='color: #1E3A8A;'>🎯 คำแนะนำสำหรับท่าน (Recommendations)</h3>", unsafe_allow_html=True)
    st.markdown("---")

    if 'results' not in st.session_state:
        st.session_state.results = {'cluster_id': 0, 'risk_score': 50.0}
    
    cluster_id = st.session_state.results.get('cluster_id', 0)
    risk_score = st.session_state.results.get('risk_score', 50.0)

    if isinstance(cluster_id, (np.ndarray, list)):
        cluster_id = int(cluster_id)
    else:
        cluster_id = int(cluster_id)

    if risk_score > 70:
        urgent_advice = "ควรเร่งสร้างวินัยทางการเงิน จัดทำบัญชีรายรับ-รายจ่ายให้ชัดเจน และลดภาระหนี้ที่ไม่จำเป็นด่วน ธนาคาร นักลงทุนและเจ้าหนี้พิจารณา 'กระแสเงินสด' ที่น่าเชื่อถือเป็นหลัก"
    elif risk_score >= 41:
        urgent_advice = "กิจการของท่านยังพอประคองตัวได้ แต่ควรระวังการใช้เงินเกินตัว ควรเริ่มจัดเตรียมเอกสารทางการเงินให้เป็นระบบ จัดเตรียมพร้อมด้านไอทีและการรองรับวิกฤติการณ์ต่าง ๆ ที่อาจเกิดขึ้น"
    else:
        urgent_advice = "กิจการมีเครือข่ายธุรกิจที่ดี บัญชีและกระแสเงินสดน่าเชื่อถือทำให้เครดิตอยู่ในเกณฑ์ยอดเยี่ยม ระบบไอทีมีความพร้อม สามารถปรับตัวกับเศรษฐกิจและวิกฤติการณ์ได้ เตรียมแผนธุรกิจเพื่อยื่นขอเงินทุนขยายกิจการได้เลย"

    recs = {
        0: { 
            "strength": "กิจการของท่านมีความเข้มแข็งด้านการตลาด การสร้างแบรนด์และภาพลักษณ์องค์กร",
            "urgent": "ควรสร้างความปลอดภัยทางเทคโนโลยี รักษาข้อมูลส่วนบุคคลของลูกค้า และกำหนดแผนรองรับวิกฤตการณ์ด่วน! ธนาคารและนักลงทุนมองว่านี่คือความเสี่ยงแฝง",
            "maintain": "รักษาฐานลูกค้าเอาไว้ให้มั่น และเสริมสร้างการตลาดออนไลน์ให้ต่อเนื่อง"
        },
        1: { 
            "strength": "กิจการของท่านมีความยืดหยุ่นและมีโอกาสในการเริ่มต้นวางระบบองค์กรที่ถูกต้อง",
            "urgent": "ควรเริ่มจัดทำบัญชีรายรับ-รายจ่ายที่ชัดเจน น่าเชื่อถือ และเสริมสร้างวินัยการเงิน แยกกระเป๋าส่วนตัวออกจากกระเป๋าของธุรกิจ ธนาคารและนักลงทุนต้องการตัวเลขที่น่าเชื่อถือ",
            "maintain": "ยึดมั่นความตั้งใจธุรกิจเอาไว้ หาความรู้เพิ่มเติมด้านการจัดการ และการวางแผนงบประมาณ"
        },
        2: { 
            "strength": "กิจการของท่านมีความพร้อมรอบด้าน ธนาคารและนักลงทุนพอใจกับกิจการลักษณะนี้",
            "urgent": "ควรหาโอกาสขยายธุรกิจให้เติบโตยิ่งขึ้น ลงทุนในนวัตกรรมเพื่อสร้างความได้เปรียบระยะยาว",
            "maintain": "รักษามาตรฐานระบบการจัดการ ส่งเสริมการตลาดและผลิตภัณฑ์ และเทคโนโลยีให้ทันสมัยอยู่เสมอ"
        }
    }

    rec = recs.get(cluster_id, recs)

    st.markdown(f"""
        <div style='background-color: #F2F2F2; padding: 15px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 25px;'>
        <p style='color: #1E3A8A; font-size: 1.1em; margin-bottom: 5px;'><b>💼 ผลลัพธ์ (จากข้อจำกัดการเข้าถึงแหล่งเงินทุน {risk_score:.1f}%)</b></p>
        <p style='margin-bottom: 0;'>{urgent_advice}</p>
        </div>
    """, unsafe_allow_html=True)

    col_rec1, col_rec2, col_rec3 = st.columns(3)

    with col_rec1:
        st.markdown(f"""
            <div style='background-color: #E2EFD9; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #2e7d32; font-size: 1.1em; margin-bottom: 10px;'><b>✅ จุดเด่น:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['strength']}</p>
            </div>
        """, unsafe_allow_html=True)

    with col_rec2:
        st.markdown(f"""
            <div style='background-color: #FFF2CC; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #c62828; font-size: 1.1em; margin-bottom: 10px;'><b>🚀 อัปเกรดด่วน:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['urgent']}</p>
            </div>
        """, unsafe_allow_html=True)

    with col_rec3:
        st.markdown(f"""
            <div style='background-color: #DEEAF6; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <p style='color: #1565c0; font-size: 1.1em; margin-bottom: 10px;'><b>🛡️ รักษาไว้:</b></p>
            <p style='font-size: 0.95em; line-height: 1.5;'>{rec['maintain']}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) 
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ถัดไป: โปรไฟล์ >", type="primary", use_container_width=True):
            navigate_to('profile')

# ==========================================
# 10. หน้า Profile & Survey
# ==========================================
def show_profile():
    scroll_to_top() 

    st.markdown("<h2 style='color:#1E3A8A; font-weight:bold;'>👤 โปรไฟล์</h2>", unsafe_allow_html=True)
    st.write("เพื่อให้งานวิจัยนี้สมบูรณ์ โปรดบันทึกข้อมูลเพื่อการอ้างอิง")
    
    with st.form("profile_form"):
        name = st.text_input("ชื่อ-นามสกุล (ระบุหรือไม่ก็ได้)")
        email = st.text_input("อีเมล (เพื่อรับผลประเมินในภายหลัง)")
        
        st.write("") 
        st.markdown("<p style='color:#FF5C8D; font-weight:bold; text-align:center;'>โปรดกดยืนยันเพื่อตอบแบบสอบถามในลำดับถัดไปครับ</p>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("ยืนยัน", type="primary", use_container_width=True)
        
    if submitted:
        st.balloons() 
        st.success("ขอบพระคุณที่ร่วมเป็นส่วนหนึ่งของงานวิจัย!")
        
        st.markdown(f"""
        <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; text-align:center; border: 1px solid #c8e6c9; margin-bottom: 20px;'>
            <h3 style='color:#2e7d32; margin-bottom:10px;'>🙏 ขอความกรุณากดลิงค์เพื่อตอบแบบสอบถามด้านล่าง</h3>
            <p style='font-size: 1.1em; color:#1b5e20;'>
                ข้อมูลของท่าน <b>{name if name else ''}</b> ได้ถูกบันทึกแล้ว<br>
                ขอบคุณครับ
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        ms_form_url = "https://forms.office.com/r/yr6x0jdH3T" 
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.link_button("📝 ทำแบบสอบถามแสดงความเห็นต่อ SME FinCheck", ms_form_url, use_container_width=True)

# ==========================================
# 11. Main Routing
# ==========================================
if st.session_state.page == 'landing':
    show_landing()
elif st.session_state.page == 'input_step1':
    show_input_step1()
elif st.session_state.page == 'input_step2':
    show_input_step2()
elif st.session_state.page == 'dashboard':
    show_dashboard()
elif st.session_state.page == 'recommendation':
    show_recommendation()
elif st.session_state.page == 'profile':
    show_profile()
