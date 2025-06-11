import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Any, Optional
import time

from config import TURBINE_ANALYSIS, ML_SETTINGS, PLOT_SETTINGS
from utils import (
    load_data, validate_turbine_data, preprocess_turbine_data,
    calculate_statistics, timer, error_handler, logger,
    create_figure, setup_plot_style
)

class TurbineAnalyzer:
    """풍력 터빈 성능 분석 클래스"""
    
    def __init__(self):
        """분석기 초기화"""
        self.data = None
        self.processed_data = None
        self.results = {
            'basic_stats': {},
            'performance_curve': {},
            'inefficiency_analysis': {},
            'correlation_analysis': {},
            'ml_models': {},
            'insights': {}
        }
        self.models = {}
        
        # 설정 로드
        self.config = TURBINE_ANALYSIS
        self.ml_config = ML_SETTINGS
        
        logger.info("TurbineAnalyzer 초기화 완료")
    
    @error_handler
    @timer
    def load_and_validate_data(self, file_path: str) -> Dict[str, Any]:
        """데이터 로딩 및 검증"""
        # 데이터 로딩
        self.data = load_data(file_path)
        
        # 데이터 검증
        validation_result = validate_turbine_data(self.data)
        
        if not validation_result['is_valid']:
            raise ValueError(f"데이터 검증 실패: {validation_result['missing_columns']}")
        
        # 전처리
        self.processed_data = preprocess_turbine_data(self.data)
        
        # 기본 정보 저장
        self.results['basic_stats']['data_info'] = {
            'original_rows': len(self.data),
            'processed_rows': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'validation': validation_result
        }
        
        logger.info(f"데이터 로딩 완료: {len(self.processed_data)}행")
        return validation_result
    
    @error_handler
    def analyze_basic_statistics(self) -> Dict[str, Any]:
        """기본 통계 분석"""
        if self.processed_data is None:
            raise ValueError("데이터를 먼저 로딩해주세요.")
        
        stats = {}
        
        # 각 수치형 컬럼별 통계
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            stats[col] = calculate_statistics(self.processed_data[col])
        
        # 발전량 특성 분석
        power_stats = {
            'zero_power_ratio': (self.processed_data['Power'] == 0).sum() / len(self.processed_data),
            'max_power': self.processed_data['Power'].max(),
            'capacity_factor': self.processed_data['Power'].mean() / self.processed_data['Power'].max()
        }
        
        # 풍속 특성 분석
        wind_stats = {
            'cut_in_violations': (self.processed_data['windspeed_100m'] < self.config['cut_in_speed']).sum(),
            'cut_out_violations': (self.processed_data['windspeed_100m'] > self.config['cut_out_speed']).sum(),
            'optimal_wind_ratio': ((self.processed_data['windspeed_100m'] >= 6) & 
                                 (self.processed_data['windspeed_100m'] <= 15)).sum() / len(self.processed_data)
        }
        
        self.results['basic_stats'].update({
            'column_statistics': stats,
            'power_analysis': power_stats,
            'wind_analysis': wind_stats
        })
        
        logger.info("기본 통계 분석 완료")
        return self.results['basic_stats']
    
    @error_handler
    def analyze_performance_curve(self) -> Dict[str, Any]:
        """성능 곡선 분석"""
        if self.processed_data is None:
            raise ValueError("데이터를 먼저 로딩해주세요.")
        
        # 풍속 구간별 분석
        max_wind_speed = self.processed_data['windspeed_100m'].max()
        bins = np.arange(0, max_wind_speed + self.config['wind_speed_bin_size'], 
                        self.config['wind_speed_bin_size'])
        
        labels = [f'{i:.1f}-{i+self.config["wind_speed_bin_size"]:.1f}' 
                 for i in bins[:-1]]
        
        self.processed_data['wind_speed_bin'] = pd.cut(
            self.processed_data['windspeed_100m'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        # 구간별 성능 통계
        power_curve_stats = self.processed_data.groupby('wind_speed_bin')['Power'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        power_curve_stats['wind_speed_center'] = bins[:-1] + self.config['wind_speed_bin_size'] / 2
        
        # 이론적 성능 곡선 (간단한 모델)
        def theoretical_power_curve(wind_speed):
            """이론적 풍력 터빈 성능 곡선"""
            cut_in = self.config['cut_in_speed']
            cut_out = self.config['cut_out_speed']
            rated_speed = 12.0  # 정격 풍속
            rated_power = self.processed_data['Power'].max()
            
            if wind_speed < cut_in:
                return 0
            elif wind_speed < rated_speed:
                # 3제곱 관계 (간단화)
                return rated_power * ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
            elif wind_speed < cut_out:
                return rated_power
            else:
                return 0
        
        power_curve_stats['theoretical_power'] = power_curve_stats['wind_speed_center'].apply(
            theoretical_power_curve
        )
        
        self.results['performance_curve'] = {
            'power_curve_data': power_curve_stats.to_dict('records'),
            'wind_speed_bins': bins.tolist(),
            'bin_size': self.config['wind_speed_bin_size']
        }
        
        logger.info("성능 곡선 분석 완료")
        return self.results['performance_curve']
    
    @error_handler
    def detect_inefficiency(self) -> Dict[str, Any]:
        """비효율 상태 탐지"""
        if 'performance_curve' not in self.results:
            self.analyze_performance_curve()
        
        # 성능 곡선 데이터 가져오기
        power_curve_df = pd.DataFrame(self.results['performance_curve']['power_curve_data'])
        
        # 데이터 병합
        merged_data = self.processed_data.merge(
            power_curve_df[['wind_speed_bin', 'mean', 'std']], 
            on='wind_speed_bin', 
            how='left'
        )
        
        # 비효율 임계값 계산
        threshold_factor = self.config['inefficiency_threshold_factor']
        merged_data['inefficiency_threshold'] = merged_data['mean'] - threshold_factor * merged_data['std']
        
        # 비효율 상태 판정
        cut_in_speed = self.config['cut_in_speed']
        merged_data['status'] = 'Normal'
        
        inefficient_mask = (
            (merged_data['Power'] < merged_data['inefficiency_threshold']) & 
            (merged_data['windspeed_100m'] > cut_in_speed) &
            (merged_data['inefficiency_threshold'].notna())
        )
        
        merged_data.loc[inefficient_mask, 'status'] = 'Inefficient'
        
        # 비효율 통계
        total_operating = (merged_data['windspeed_100m'] > cut_in_speed).sum()
        inefficient_count = (merged_data['status'] == 'Inefficient').sum()
        inefficient_percentage = (inefficient_count / total_operating * 100) if total_operating > 0 else 0
        
        # 시간대별 비효율 분석
        if hasattr(merged_data.index, 'hour'):
            merged_data['hour'] = merged_data.index.hour
        else:
            # 인덱스가 datetime이 아닌 경우 Time 컬럼 활용
            if 'Time' in merged_data.columns:
                merged_data['hour'] = pd.to_datetime(merged_data['Time']).dt.hour
            else:
                # 시간 정보가 없는 경우 더미 시간 생성
                merged_data['hour'] = merged_data.reset_index().index % 24
        
        hourly_inefficiency = merged_data.groupby('hour')['status'].apply(
            lambda x: (x == 'Inefficient').sum() / len(x) * 100 if len(x) > 0 else 0
        ).to_dict()
        
        self.processed_data = merged_data  # 업데이트된 데이터 저장
        
        self.results['inefficiency_analysis'] = {
            'total_operating_points': int(total_operating),
            'inefficient_points': int(inefficient_count),
            'inefficiency_percentage': round(inefficient_percentage, 2),
            'threshold_factor': threshold_factor,
            'hourly_inefficiency': hourly_inefficiency,
            'inefficient_data_summary': {
                'avg_power_loss': merged_data[inefficient_mask]['Power'].mean() if inefficient_count > 0 else 0,
                'avg_wind_speed': merged_data[inefficient_mask]['windspeed_100m'].mean() if inefficient_count > 0 else 0
            }
        }
        
        logger.info(f"비효율 분석 완료: {inefficient_percentage:.1f}% 비효율 상태 탐지")
        return self.results['inefficiency_analysis']
    
    @error_handler
    def analyze_correlations(self) -> Dict[str, Any]:
        """상관관계 분석"""
        if self.processed_data is None:
            raise ValueError("데이터를 먼저 로딩해주세요.")
        
        # 수치형 컬럼만 선택
        numeric_columns = self.processed_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.processed_data[numeric_columns].corr()
        
        # Power와의 상관관계
        power_correlations = correlation_matrix['Power'].drop('Power').sort_values(
            key=abs, ascending=False
        )
        
        # 강한 상관관계 (|r| > 0.5)
        strong_correlations = power_correlations[abs(power_correlations) > 0.5]
        
        self.results['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'power_correlations': power_correlations.to_dict(),
            'strong_correlations': strong_correlations.to_dict(),
            'top_positive_correlation': power_correlations.idxmax(),
            'top_negative_correlation': power_correlations.idxmin()
        }
        
        logger.info("상관관계 분석 완료")
        return self.results['correlation_analysis']
    
    @error_handler
    @timer
    def train_prediction_models(self) -> Dict[str, Any]:
        """머신러닝 모델 훈련"""
        if self.processed_data is None:
            raise ValueError("데이터를 먼저 로딩해주세요.")
        
        # 특성 및 타겟 설정
        feature_columns = [
            'windspeed_100m', 'winddirection_100m', 'temperature_2m', 
            'relativehumidity_2m', 'dewpoint_2m'
        ]
        
        # 결측치가 없는 데이터만 사용
        clean_data = self.processed_data[feature_columns + ['Power']].dropna()
        
        X = clean_data[feature_columns]
        y = clean_data['Power']
        
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.ml_config['test_size'], 
            random_state=self.ml_config['random_state']
        )
        
        # 모델 정의
        models = {
            #'Linear Regression': Pipeline([
            #    ('scaler', StandardScaler()), 
            #    ('model', LinearRegression())
            #]),
            #'SVR': Pipeline([
            #    ('scaler', StandardScaler()), 
            #    ('model', SVR())
            #]),
            'Random Forest': RandomForestRegressor(
                **self.ml_config['models']['random_forest']
            ),
        }
        
        # 모델 훈련 및 평가
        model_results = []
        
        for name, model in models.items():
            start_time = time.time()
            
            # 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 평가
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            model_result = {
                'model_name': name,
                'mae': mae,
                'r2_score': r2,
                'training_time': training_time,
                'test_size': len(y_test)
            }
            
            model_results.append(model_result)
            self.models[name] = model  # 모델 저장
            
            logger.info(f"{name} 훈련 완료 - MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # 최고 성능 모델 선택
        best_model = max(model_results, key=lambda x: x['r2_score'])
        
        # 특성 중요도 (Random Forest의 경우)
        feature_importance = {}
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importance_scores = rf_model.feature_importances_
            feature_importance = dict(zip(feature_columns, importance_scores))
        
        self.results['ml_models'] = {
            'model_results': model_results,
            'best_model': best_model,
            'feature_importance': feature_importance,
            'feature_columns': feature_columns,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'total_size': len(clean_data)
            }
        }
        
        logger.info(f"머신러닝 모델 훈련 완료. 최고 성능: {best_model['model_name']} (R²: {best_model['r2_score']:.4f})")
        return self.results['ml_models']
    
    @error_handler
    def generate_comprehensive_insights(self) -> Dict[str, Any]:
        """종합 인사이트 생성"""
        insights = {
            'data_overview': {},
            'performance_insights': {},
            'efficiency_insights': {},
            'predictive_insights': {},
            'recommendations': []
        }
        
        # 데이터 개요
        if 'basic_stats' in self.results:
            basic_stats = self.results['basic_stats']
            insights['data_overview'] = {
                'total_records': basic_stats.get('data_info', {}).get('processed_rows', 0),
                'power_capacity_factor': basic_stats.get('power_analysis', {}).get('capacity_factor', 0),
                'zero_power_ratio': basic_stats.get('power_analysis', {}).get('zero_power_ratio', 0)
            }
        
        # 성능 인사이트
        if 'performance_curve' in self.results:
            perf_data = pd.DataFrame(self.results['performance_curve']['power_curve_data'])
            if not perf_data.empty:
                optimal_wind_bin = perf_data.loc[perf_data['mean'].idxmax()]
                insights['performance_insights'] = {
                    'optimal_wind_speed_range': optimal_wind_bin['wind_speed_bin'],
                    'max_average_power': optimal_wind_bin['mean'],
                    'power_variability': optimal_wind_bin['std']
                }
        
        # 효율성 인사이트
        if 'inefficiency_analysis' in self.results:
            ineff_data = self.results['inefficiency_analysis']
            insights['efficiency_insights'] = {
                'inefficiency_percentage': ineff_data.get('inefficiency_percentage', 0),
                'total_operating_hours': ineff_data.get('total_operating_points', 0),
                'potential_improvement': ineff_data.get('inefficient_data_summary', {})
            }
        
        # 예측 모델 인사이트
        if 'ml_models' in self.results:
            ml_data = self.results['ml_models']
            insights['predictive_insights'] = {
                'best_model_performance': ml_data.get('best_model', {}),
                'key_factors': ml_data.get('feature_importance', {}),
                'prediction_reliability': ml_data.get('best_model', {}).get('r2_score', 0)
            }
        
        # 권장사항 생성
        recommendations = []
        
        # 비효율성 기반 권장사항
        if insights['efficiency_insights']['inefficiency_percentage'] > 10:
            recommendations.append("비효율 상태가 10% 이상 발생하고 있어 정비가 필요합니다.")
        
        # 용량계수 기반 권장사항
        capacity_factor = insights['data_overview'].get('power_capacity_factor', 0)
        if capacity_factor < 0.3:
            recommendations.append("용량계수가 낮습니다. 입지 재검토나 터빈 업그레이드를 고려하세요.")
        
        # 예측 모델 기반 권장사항
        if insights['predictive_insights']['prediction_reliability'] > 0.8:
            recommendations.append("예측 모델 성능이 우수하여 예측 기반 운영 최적화가 가능합니다.")
        
        insights['recommendations'] = recommendations
        
        self.results['insights'] = insights
        logger.info("종합 인사이트 생성 완료")
        return insights
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        return {
            'timestamp': pd.Timestamp.now(),
            'analysis_completed': list(self.results.keys()),
            'data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'model_count': len(self.models),
            'results': self.results
        }
    
    def run_full_analysis(self, file_path: str) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info("전체 분석 시작")
        
        # 단계별 분석 실행
        self.load_and_validate_data(file_path)
        self.analyze_basic_statistics()
        self.analyze_performance_curve()
        self.detect_inefficiency()
        self.analyze_correlations()
        self.train_prediction_models()
        self.generate_comprehensive_insights()
        
        logger.info("전체 분석 완료")
        return self.get_analysis_summary()