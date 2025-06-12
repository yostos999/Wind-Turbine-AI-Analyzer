import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import traceback

from config import get_current_llm_config, LLM_PROVIDER
from utils import logger, error_handler

# =============================================================================
# 프롬프트 템플릿
# =============================================================================

PROMPT_TEMPLATES = {
    "data_summary": """
당신은 풍력 발전소의 설비 전문가입니다. 다음 데이터 분석 결과를 바탕으로 운영 현황을 요약해주세요.

## 분석 데이터:
{analysis_data}

다음 관점에서 요약해주세요:
1. **데이터 개요**: 분석 기간, 데이터 품질, 주요 특징
2. **발전 성능**: 용량계수, 가동률, 평균 발전량
3. **운영 효율**: 정상 운영 비율, 주요 성능 지표
4. **핵심 발견사항**: 가장 중요한 3가지 포인트

전문적이고 간결하게 작성해주세요.
""",

    "performance_analysis": """
당신은 풍력 터빈 성능 분석 전문가입니다. 다음 성능 데이터를 분석하여 인사이트를 제공해주세요.

## 성능 분석 데이터:
{analysis_data}

다음 항목에 대해 상세히 분석해주세요:
1. **성능 곡선 분석**: 풍속별 발전 효율, 최적 운영 구간
2. **비효율 구간 분석**: 성능 저하 원인, 발생 패턴
3. **환경 요인 영향**: 온도, 습도, 풍향이 성능에 미치는 영향
4. **상관관계 분석**: 주요 변수 간의 관계 해석

설비팀이 이해하기 쉽게 설명해주세요.
""",

    "efficiency_diagnosis": """
당신은 설비 효율성 진단 전문가입니다. 다음 비효율 분석 결과를 바탕으로 문제점을 진단해주세요.

## 비효율 분석 데이터:
{analysis_data}

다음 관점에서 진단해주세요:
1. **비효율 현황**: 전체 대비 비효율 비율, 심각도 평가
2. **패턴 분석**: 시간대별, 풍속별 비효율 발생 패턴
3. **근본 원인**: 예상되는 비효율 발생 원인들
4. **영향도 평가**: 발전량 손실, 수익성 영향

정비팀이 우선순위를 정할 수 있도록 구체적으로 분석해주세요.
""",

    "improvement_recommendations": """
당신은 풍력 발전소 운영 최적화 컨설턴트입니다. 다음 종합 분석 결과를 바탕으로 개선 방안을 제시해주세요.

## 종합 분석 결과:
{analysis_data}

다음 형태로 개선 방안을 제시해주세요:
1. **즉시 실행 가능한 개선안** (1-2주 내 실행)
2. **단기 개선안** (1-3개월 내 실행)  
3. **중장기 개선안** (6개월-1년 내 실행)
4. **투자 대비 효과 분석**: 각 개선안의 예상 효과

각 개선안에는 다음을 포함해주세요:
- 구체적인 실행 방법
- 예상 비용 및 기간
- 기대 효과 (발전량 증가, 비용 절감 등)
- 실행 시 주의사항

실무진이 바로 실행할 수 있도록 구체적이고 실용적으로 작성해주세요.
""",

    "comprehensive_report": """
당신은 풍력 발전소 운영 분석 전문가입니다. 다음 전체 분석 결과를 바탕으로 경영진 보고용 종합 리포트를 작성해주세요.

## 전체 분석 데이터:
{analysis_data}

다음 구조로 리포트를 작성해주세요:

### 📊 운영 현황 요약
- 핵심 KPI (용량계수, 가동률, 발전량)

### 🔍 주요 발견사항
- 성능 우수 구간과 문제 구간
- 비효율 발생 현황 및 원인

### 💡 개선 기회
- 즉시 개선 가능한 항목들
- 예상 개선 효과 (정량적)

### 🎯 권장 액션
- 우선순위별 실행 계획
- 필요 자원 및 예산

### 📈 향후 전망
- 예측 모델 기반 성과 전망
- 지속적 모니터링 포인트

경영진과 기술진 모두가 이해할 수 있도록 명확하고 간결하게 작성해주세요.
"""
}

# =============================================================================
# 추상 기본 클래스
# =============================================================================

class LLMInterface(ABC):
    """LLM 인터페이스 추상 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """모델 초기화 (각 LLM별 구현)"""
        pass
    
    @abstractmethod
    def _generate_text(self, prompt: str) -> str:
        """텍스트 생성 (각 LLM별 구현)"""
        pass
    
    def generate_insight(self, analysis_data: Dict[str, Any], 
                        insight_type: str = "comprehensive_report") -> Dict[str, Any]:
        """인사이트 생성"""
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(analysis_data, insight_type)
            
            # LLM 호출
            response = self._generate_text(prompt)
            
            return {
                "success": True,
                "insight_type": insight_type,
                "content": response,
                "timestamp": time.time(),
                "model_info": {
                    "provider": LLM_PROVIDER,
                    "model_name": self.config.get("model_name", "unknown")
                }
            }
            
        except Exception as e:
            logger.error(f"인사이트 생성 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "insight_type": insight_type,
                "timestamp": time.time()
            }
    
    def _create_prompt(self, analysis_data: Dict[str, Any], insight_type: str) -> str:
        """프롬프트 생성"""
        if insight_type not in PROMPT_TEMPLATES:
            raise ValueError(f"지원하지 않는 인사이트 타입: {insight_type}")
        
        # 분석 데이터를 JSON 형태로 정리
        data_summary = self._format_analysis_data(analysis_data)
        
        # 프롬프트 템플릿에 데이터 삽입
        prompt = PROMPT_TEMPLATES[insight_type].format(analysis_data=data_summary)
        
        return prompt
    
    def _format_analysis_data(self, data: Dict[str, Any]) -> str:
        """분석 데이터를 LLM이 이해하기 쉬운 형태로 포맷팅"""
        formatted_sections = []
        
        # 기본 통계
        if "basic_stats" in data:
            basic_info = data["basic_stats"]
            formatted_sections.append(f"""
### 기본 데이터 정보:
- 총 데이터 포인트: {basic_info.get('data_info', {}).get('processed_rows', 'N/A')}개
- 용량계수: {basic_info.get('power_analysis', {}).get('capacity_factor', 0):.3f}
- 무발전 비율: {basic_info.get('power_analysis', {}).get('zero_power_ratio', 0):.3f}
- 최대 발전량: {basic_info.get('power_analysis', {}).get('max_power', 'N/A')} kW
""")
        
        # 성능 곡선
        if "performance_curve" in data:
            perf_data = data["performance_curve"]
            formatted_sections.append(f"""
### 성능 곡선 분석:
- 풍속 구간 크기: {perf_data.get('bin_size', 'N/A')} m/s
- 분석된 풍속 구간 수: {len(perf_data.get('power_curve_data', []))}개
""")
        
        # 비효율 분석
        if "inefficiency_analysis" in data:
            ineff_data = data["inefficiency_analysis"]
            formatted_sections.append(f"""
### 비효율 분석:
- 비효율 발생률: {ineff_data.get('inefficiency_percentage', 0):.2f}%
- 총 운영 데이터: {ineff_data.get('total_operating_points', 'N/A')}개
- 비효율 데이터: {ineff_data.get('inefficient_points', 'N/A')}개
- 평균 발전 손실: {ineff_data.get('inefficient_data_summary', {}).get('avg_power_loss', 'N/A')} kW
""")
        
        # 상관관계 분석
        if "correlation_analysis" in data:
            corr_data = data["correlation_analysis"]
            strong_corr = corr_data.get("strong_correlations", {})
            formatted_sections.append(f"""
### 상관관계 분석:
- 발전량과 강한 상관관계를 보이는 변수: {len(strong_corr)}개
- 최고 양의 상관관계: {corr_data.get('top_positive_correlation', 'N/A')}
- 최고 음의 상관관계: {corr_data.get('top_negative_correlation', 'N/A')}
""")
        
        # 머신러닝 모델
        if "ml_models" in data:
            ml_data = data["ml_models"]
            best_model = ml_data.get("best_model", {})
            formatted_sections.append(f"""
### 예측 모델 성능:
- 최고 성능 모델: {best_model.get('model_name', 'N/A')}
- R² 점수: {best_model.get('r2_score', 0):.4f}
- 평균 절대 오차: {best_model.get('mae', 0):.4f} kW
- 훈련 데이터: {ml_data.get('data_split', {}).get('train_size', 'N/A')}개
""")
        
        # 종합 인사이트
        if "insights" in data:
            insights_data = data["insights"]
            formatted_sections.append(f"""
### 종합 인사이트:
- 권장사항 수: {len(insights_data.get('recommendations', []))}개
- 예측 신뢰도: {insights_data.get('predictive_insights', {}).get('prediction_reliability', 0):.3f}
""")
        
        return "\n".join(formatted_sections)

# =============================================================================
# Gemini 구현
# =============================================================================

class GeminiInterface(LLMInterface):
    """Google Gemini API 인터페이스"""
    
    def _initialize_model(self):
        """Gemini 모델 초기화"""
        try:
            import google.generativeai as genai
            
            # API 키 설정
            genai.configure(api_key=self.config["api_key"])
            
            # 모델 생성
            generation_config = {
                "temperature": self.config.get("temperature", 0.3),
                "top_p": self.config.get("top_p", 0.8),
                "top_k": self.config.get("top_k", 40),
                "max_output_tokens": self.config.get("max_tokens", 2048),
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.config["model_name"],
                generation_config=generation_config
            )
            
            logger.info(f"Gemini 모델 초기화 완료: {self.config['model_name']}")
            
        except ImportError:
            raise ImportError("google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai")
        except Exception as e:
            logger.error(f"Gemini 모델 초기화 실패: {str(e)}")
            raise
    
    def _generate_text(self, prompt: str) -> str:
        """Gemini API로 텍스트 생성"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text
                else:
                    raise ValueError("API 응답이 비어있습니다.")
                    
            except Exception as e:
                logger.warning(f"Gemini API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 지수 백오프
                else:
                    raise

# =============================================================================
# OpenAI 구현 (추후 확장용)
# =============================================================================

class OpenAIInterface(LLMInterface):
    """OpenAI API 인터페이스 (추후 구현)"""
    
    def _initialize_model(self):
        # TODO: OpenAI API 초기화
        raise NotImplementedError("OpenAI 인터페이스는 아직 구현되지 않았습니다.")
    
    def _generate_text(self, prompt: str) -> str:
        # TODO: OpenAI API 호출
        raise NotImplementedError("OpenAI 인터페이스는 아직 구현되지 않았습니다.")

# =============================================================================
# Claude 구현 (추후 확장용)
# =============================================================================

class ClaudeInterface(LLMInterface):
    """Anthropic Claude API 인터페이스 (추후 구현)"""
    
    def _initialize_model(self):
        # TODO: Claude API 초기화
        raise NotImplementedError("Claude 인터페이스는 아직 구현되지 않았습니다.")
    
    def _generate_text(self, prompt: str) -> str:
        # TODO: Claude API 호출
        raise NotImplementedError("Claude 인터페이스는 아직 구현되지 않았습니다.")

# =============================================================================
# 팩토리 함수
# =============================================================================

def create_llm_interface() -> LLMInterface:
    """현재 설정에 따른 LLM 인터페이스 생성"""
    config = get_current_llm_config()
    
    if LLM_PROVIDER == "gemini":
        return GeminiInterface(config)
    elif LLM_PROVIDER == "openai":
        return OpenAIInterface(config)
    elif LLM_PROVIDER == "claude":
        return ClaudeInterface(config)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자: {LLM_PROVIDER}")

# =============================================================================
# 편의 함수
# =============================================================================

class InsightGenerator:
    """인사이트 생성 편의 클래스"""
    
    def __init__(self):
        self.llm = create_llm_interface()
    
    @error_handler
    def generate_data_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 요약 생성"""
        return self.llm.generate_insight(analysis_results, "data_summary")
    
    @error_handler
    def generate_performance_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """성능 분석 생성"""
        return self.llm.generate_insight(analysis_results, "performance_analysis")
    
    @error_handler
    def generate_efficiency_diagnosis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """효율성 진단 생성"""
        return self.llm.generate_insight(analysis_results, "efficiency_diagnosis")
    
    @error_handler
    def generate_improvement_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """개선 방안 생성"""
        return self.llm.generate_insight(analysis_results, "improvement_recommendations")
    
    @error_handler
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 리포트 생성"""
        return self.llm.generate_insight(analysis_results, "comprehensive_report")
    
    def generate_all_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """모든 타입의 인사이트 생성"""
        all_insights = {}
        
        insight_types = [
            ("data_summary", self.generate_data_summary),
            ("performance_analysis", self.generate_performance_analysis),
            ("efficiency_diagnosis", self.generate_efficiency_diagnosis),
            ("improvement_recommendations", self.generate_improvement_recommendations),
            ("comprehensive_report", self.generate_comprehensive_report)
        ]
        
        for insight_type, generator_func in insight_types:
            try:
                result = generator_func(analysis_results)
                all_insights[insight_type] = result
                logger.info(f"{insight_type} 인사이트 생성 완료")
            except Exception as e:
                logger.error(f"{insight_type} 인사이트 생성 실패: {str(e)}")
                all_insights[insight_type] = {
                    "success": False,
                    "error": str(e),
                    "insight_type": insight_type
                }
        
        return all_insights

if __name__ == "__main__":
    # 테스트 코드
    try:
        generator = InsightGenerator()
        print(f"✅ LLM 인터페이스 초기화 성공: {LLM_PROVIDER}")
        print(f"모델: {generator.llm.config['model_name']}")
    except Exception as e:
        print(f"❌ LLM 인터페이스 초기화 실패: {str(e)}")
        print("API 키와 설정을 확인해주세요.")