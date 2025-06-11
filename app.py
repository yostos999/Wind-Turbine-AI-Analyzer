import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import base64
from datetime import datetime
import traceback

# 로컬 모듈 import
from config import APP_CONFIG, FILE_SETTINGS, validate_config, get_available_providers, switch_llm_provider
from data_analyzer import TurbineAnalyzer
from llm_interface import InsightGenerator
from utils import logger, setup_plot_style

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title=APP_CONFIG["page_title"],
    page_icon=APP_CONFIG["page_icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state=APP_CONFIG["initial_sidebar_state"]
)

# 스타일 설정
setup_plot_style()

# =============================================================================
# 유틸리티 함수
# =============================================================================

def download_button(data, filename, label):
    """다운로드 버튼 생성"""
    if isinstance(data, dict):
        data = json.dumps(data, indent=2, ensure_ascii=False)
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    return href

def display_metrics(data, title):
    """메트릭 카드 표시"""
    st.subheader(title)
    
    if isinstance(data, dict):
        cols = st.columns(len(data))
        for i, (key, value) in enumerate(data.items()):
            with cols[i]:
                if isinstance(value, float):
                    st.metric(key, f"{value:.3f}")
                else:
                    st.metric(key, str(value))

def create_correlation_heatmap(corr_matrix):
    """상관관계 히트맵 생성"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="변수 간 상관관계",
        width=600,
        height=500
    )
    
    return fig

def create_power_curve_plot(power_curve_data):
    """성능 곡선 플롯 생성"""
    df = pd.DataFrame(power_curve_data)
    
    fig = go.Figure()
    
    # 평균 성능 곡선
    fig.add_trace(go.Scatter(
        x=df['wind_speed_center'],
        y=df['mean'],
        mode='lines+markers',
        name='실제 평균 성능',
        line=dict(color='blue', width=3)
    ))
    
    # 표준편차 범위
    fig.add_trace(go.Scatter(
        x=df['wind_speed_center'],
        y=df['mean'] + 2*df['std'],
        mode='lines',
        line=dict(color='gray', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['wind_speed_center'],
        y=df['mean'] - 2*df['std'],
        mode='lines',
        line=dict(color='gray', width=0),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.3)',
        name='정상 운영 범위 (±2σ)',
        hoverinfo='skip'
    ))
    
    # 이론적 성능 곡선 (있는 경우)
    if 'theoretical_power' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['wind_speed_center'],
            y=df['theoretical_power'],
            mode='lines',
            name='이론적 성능',
            line=dict(color='red', dash='dash', width=2)
        ))
    
    fig.update_layout(
        title="풍력 터빈 성능 곡선",
        xaxis_title="풍속 (m/s)",
        yaxis_title="발전량 (kW)",
        hovermode='x unified'
    )
    
    return fig

# =============================================================================
# 세션 상태 초기화
# =============================================================================

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# =============================================================================
# 메인 앱
# =============================================================================

def main():
    # 헤더
    st.title("🔧 AI 설비 성능 분석가")
    st.markdown("""
    풍력 터빈 성능 데이터를 분석하고 AI 기반 인사이트를 제공하는 시스템입니다.
    데이터를 업로드하여 종합적인 성능 분석과 개선 방안을 확인하세요.
    """)
    
    # 설정 검증
    config_errors = validate_config()
    if config_errors:
        st.error("⚠️ 설정 오류가 발견되었습니다:")
        for error in config_errors:
            st.error(f"• {error}")
        st.info("config.py 파일과 .env 파일을 확인해주세요.")
        return
    
    # 사이드바
    setup_sidebar()
    
    # 메인 콘텐츠
    if st.session_state.analysis_complete:
        display_analysis_results()
    else:
        display_welcome_screen()

def setup_sidebar():
    """사이드바 설정"""
    st.sidebar.header("⚙️ 설정")
    
    # LLM 제공자 선택
    available_providers = get_available_providers()
    if available_providers:
        selected_provider = st.sidebar.selectbox(
            "LLM 제공자 선택",
            available_providers,
            help="AI 인사이트 생성에 사용할 LLM을 선택하세요."
        )
        
        if st.sidebar.button("LLM 제공자 변경"):
            if switch_llm_provider(selected_provider):
                st.sidebar.success(f"✅ {selected_provider}로 변경되었습니다.")
                st.rerun()
    else:
        st.sidebar.error("❌ 사용 가능한 LLM 제공자가 없습니다.")
    
    st.sidebar.markdown("---")
    
    # 파일 업로드
    st.sidebar.header("📁 데이터 업로드")
    
    uploaded_file = st.sidebar.file_uploader(
        "풍력 터빈 데이터 파일",
        type=["csv", "xlsx", "xls"],
        help=f"최대 {FILE_SETTINGS['max_file_size_mb']}MB까지 업로드 가능합니다."
    )
    
    # 샘플 데이터 사용 옵션
    use_sample = st.sidebar.checkbox(
        "샘플 데이터 사용",
        value=True,
        help="기본 제공되는 샘플 데이터를 사용합니다."
    )
    
    # 분석 실행
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        if uploaded_file is not None or use_sample:
            run_analysis(uploaded_file, use_sample)
        else:
            st.sidebar.error("❌ 파일을 업로드하거나 샘플 데이터를 선택해주세요.")
    
    # 초기화 버튼
    if st.sidebar.button("🔄 초기화"):
        reset_session_state()
        st.rerun()

def reset_session_state():
    """세션 상태 초기화"""
    st.session_state.analyzer = None
    st.session_state.analysis_results = None
    st.session_state.ai_insights = None
    st.session_state.analysis_complete = False

def run_analysis(uploaded_file, use_sample):
    """분석 실행"""
    try:
        with st.sidebar:
            with st.spinner("분석을 진행 중입니다..."):
                # 데이터 경로 결정
                if use_sample:
                    file_path = FILE_SETTINGS["default_data_path"]
                else:
                    # 업로드된 파일을 임시 저장
                    file_path = f"temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # 분석 실행
                analyzer = TurbineAnalyzer()
                analysis_results = analyzer.run_full_analysis(file_path)
                
                # 세션 상태 업데이트
                st.session_state.analyzer = analyzer
                st.session_state.analysis_results = analysis_results
                st.session_state.analysis_complete = True
                
                # 임시 파일 정리
                if not use_sample and file_path.startswith("temp_"):
                    import os
                    os.remove(file_path)
        
        st.sidebar.success("✅ 분석이 완료되었습니다!")
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"분석 실행 오류: {traceback.format_exc()}")

def display_welcome_screen():
    """시작 화면 표시"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 시작하기")
        st.markdown("""
        1. **데이터 준비**: 사이드바에서 파일을 업로드하거나 샘플 데이터를 선택하세요.
        2. **분석 실행**: '분석 시작' 버튼을 클릭하여 종합 분석을 시작하세요.
        3. **결과 확인**: 분석 완료 후 각 탭에서 상세 결과를 확인할 수 있습니다.
        4. **AI 인사이트**: LLM 기반 인사이트와 개선 방안을 확인하세요.
        """)
        
        st.markdown("### 📋 필요한 데이터 컬럼")
        required_columns = [
            "Time - 시간 정보",
            "Power - 발전량 (kW)",
            "windspeed_100m - 100m 높이 풍속 (m/s)",
            "winddirection_100m - 100m 높이 풍향 (도)",
            "temperature_2m - 2m 높이 온도 (°C)",
            "relativehumidity_2m - 2m 높이 상대습도 (%)",
            "dewpoint_2m - 2m 높이 이슬점 (°C)"
        ]
        
        for col in required_columns:
            st.markdown(f"• {col}")

def display_analysis_results():
    """분석 결과 표시"""
    if st.session_state.analysis_results is None:
        st.error("❌ 분석 결과가 없습니다.")
        return
    
    results = st.session_state.analysis_results['results']
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 데이터 개요", 
        "⚡ 성능 분석", 
        "🔍 효율성 진단", 
        "🤖 AI 인사이트", 
        "📋 종합 리포트"
    ])
    
    with tab1:
        display_data_overview(results)
    
    with tab2:
        display_performance_analysis(results)
    
    with tab3:
        display_efficiency_analysis(results)
    
    with tab4:
        display_ai_insights(results)
    
    with tab5:
        display_comprehensive_report(results)

def display_data_overview(results):
    """데이터 개요 표시"""
    st.header("📊 데이터 개요")
    
    if 'basic_stats' in results:
        basic_stats = results['basic_stats']
        
        # 데이터 정보
        st.subheader("📋 데이터 정보")
        data_info = basic_stats.get('data_info', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 데이터", f"{data_info.get('processed_rows', 0):,}개")
        with col2:
            st.metric("분석 컬럼", f"{len(data_info.get('columns', []))}개")
        with col3:
            power_analysis = basic_stats.get('power_analysis', {})
            st.metric("용량계수", f"{power_analysis.get('capacity_factor', 0):.3f}")
        with col4:
            st.metric("최대 발전량", f"{power_analysis.get('max_power', 0):.1f} kW")
        
        # 기본 통계
        st.subheader("📈 주요 통계")
        if 'column_statistics' in basic_stats:
            stats_df = pd.DataFrame(basic_stats['column_statistics']).T
            stats_df = stats_df.round(3)
            st.dataframe(stats_df, use_container_width=True)

def display_performance_analysis(results):
    """성능 분석 표시"""
    st.header("⚡ 성능 분석")
    
    # 성능 곡선
    if 'performance_curve' in results:
        st.subheader("🌪️ 풍력 터빈 성능 곡선")
        power_curve_data = results['performance_curve']['power_curve_data']
        
        if power_curve_data:
            fig = create_power_curve_plot(power_curve_data)
            st.plotly_chart(fig, use_container_width=True)
    
    # 상관관계 분석
    if 'correlation_analysis' in results:
        st.subheader("🔗 변수 간 상관관계")
        corr_data = results['correlation_analysis']
        
        if 'correlation_matrix' in corr_data:
            corr_matrix = pd.DataFrame(corr_data['correlation_matrix'])
            fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
        
        # 발전량과의 상관관계
        if 'power_correlations' in corr_data:
            st.subheader("⚡ 발전량과의 상관관계")
            power_corr = pd.Series(corr_data['power_correlations']).sort_values(key=abs, ascending=False)
            
            fig = go.Figure(go.Bar(
                x=power_corr.values,
                y=power_corr.index,
                orientation='h',
                marker_color=['red' if x < 0 else 'blue' for x in power_corr.values]
            ))
            
            fig.update_layout(
                title="발전량과 각 변수의 상관계수",
                xaxis_title="상관계수",
                yaxis_title="변수"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_efficiency_analysis(results):
    """효율성 분석 표시"""
    st.header("🔍 효율성 진단")
    
    if 'inefficiency_analysis' in results:
        ineff_data = results['inefficiency_analysis']
        
        # 핵심 지표
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("비효율 발생률", f"{ineff_data.get('inefficiency_percentage', 0):.2f}%")
        with col2:
            st.metric("총 운영 시간", f"{ineff_data.get('total_operating_points', 0):,}시간")
        with col3:
            st.metric("비효율 시간", f"{ineff_data.get('inefficient_points', 0):,}시간")
        with col4:
            summary = ineff_data.get('inefficient_data_summary', {})
            st.metric("평균 손실량", f"{summary.get('avg_power_loss', 0):.1f} kW")
        
        # 시간대별 비효율 패턴
        if 'hourly_inefficiency' in ineff_data:
            st.subheader("⏰ 시간대별 비효율 패턴")
            hourly_data = ineff_data['hourly_inefficiency']
            
            hours = list(hourly_data.keys())
            inefficiency_rates = list(hourly_data.values())
            
            fig = go.Figure(go.Bar(
                x=hours,
                y=inefficiency_rates,
                name='비효율 발생률 (%)',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="시간대별 비효율 발생률",
                xaxis_title="시간",
                yaxis_title="비효율 발생률 (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_ai_insights(results):
    """AI 인사이트 표시"""
    st.header("🤖 AI 인사이트")
    
    # AI 인사이트 생성 버튼
    if st.button("✨ AI 인사이트 생성", type="primary"):
        generate_ai_insights(results)
    
    # 생성된 인사이트 표시
    if st.session_state.ai_insights:
        insights = st.session_state.ai_insights
        
        insight_tabs = st.tabs([
            "📋 데이터 요약",
            "⚡ 성능 분석", 
            "🔍 효율성 진단",
            "💡 개선 방안"
        ])
        
        tab_mapping = [
            ("data_summary", "📋 데이터 요약"),
            ("performance_analysis", "⚡ 성능 분석"),
            ("efficiency_diagnosis", "🔍 효율성 진단"),
            ("improvement_recommendations", "💡 개선 방안")
        ]
        
        for i, (insight_type, tab_name) in enumerate(tab_mapping):
            with insight_tabs[i]:
                if insight_type in insights:
                    insight_data = insights[insight_type]
                    if insight_data.get('success', False):
                        st.markdown(insight_data['content'])
                    else:
                        st.error(f"❌ {tab_name} 생성 실패: {insight_data.get('error', '알 수 없는 오류')}")
                else:
                    st.info("💡 'AI 인사이트 생성' 버튼을 클릭하여 인사이트를 생성하세요.")

def generate_ai_insights(results):
    """AI 인사이트 생성"""
    try:
        with st.spinner("🤖 AI가 인사이트를 생성하고 있습니다..."):
            generator = InsightGenerator()
            insights = generator.generate_all_insights(results)
            st.session_state.ai_insights = insights
        
        st.success("✅ AI 인사이트가 생성되었습니다!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ AI 인사이트 생성 중 오류: {str(e)}")
        logger.error(f"AI 인사이트 생성 오류: {traceback.format_exc()}")

def display_comprehensive_report(results):
    """종합 리포트 표시"""
    st.header("📋 종합 리포트")
    
    # 종합 리포트 생성
    if st.button("📊 종합 리포트 생성", type="primary"):
        try:
            with st.spinner("📊 종합 리포트를 생성하고 있습니다..."):
                generator = InsightGenerator()
                report = generator.generate_comprehensive_report(results)
                
                if report.get('success', False):
                    st.markdown("### 🎯 AI 생성 종합 리포트")
                    st.markdown(report['content'])
                    
                    # 다운로드 버튼
                    report_text = f"""
# 풍력 터빈 성능 분석 종합 리포트
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report['content']}

---
분석 도구: AI 설비 성능 분석가
LLM 모델: {report.get('model_info', {}).get('model_name', 'Unknown')}
                    """
                    
                    st.download_button(
                        label="📥 리포트 다운로드",
                        data=report_text,
                        file_name=f"turbine_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error(f"❌ 리포트 생성 실패: {report.get('error', '알 수 없는 오류')}")
                    
        except Exception as e:
            st.error(f"❌ 종합 리포트 생성 중 오류: {str(e)}")
    
    # 기존 인사이트가 있으면 표시
    if st.session_state.ai_insights and 'comprehensive_report' in st.session_state.ai_insights:
        report = st.session_state.ai_insights['comprehensive_report']
        if report.get('success', False):
            st.markdown("### 📊 기존 생성된 종합 리포트")
            st.markdown(report['content'])

# =============================================================================
# 앱 실행
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {str(e)}")
        logger.error(f"앱 실행 오류: {traceback.format_exc()}")
        
        with st.expander("🔧 문제 해결 도움말"):
            st.markdown("""
            **일반적인 해결 방법:**
            1. 페이지를 새로고침하세요.
            2. .env 파일의 API 키가 올바른지 확인하세요.
            3. requirements.txt의 모든 패키지가 설치되었는지 확인하세요.
            4. 데이터 파일 형식이 올바른지 확인하세요.
            
            **개발자 정보:**
            ```
            {traceback.format_exc()}
            ```
            """)