from advanced_trading_algorithms import AdvancedTradingAlgorithms
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class SignalIntegration:
    def __init__(self):
        self.advanced_algo = AdvancedTradingAlgorithms()
        self.timeframes = {
            '1d': 1,    # Daily
            '2w': 14,   # 2 weeks
            '1m': 30,   # 1 month
            '2m': 60,   # 2 months
            '3m': 90    # 3 months
        }
        
    def analyze_timeframe(self, prices: pd.Series, volumes: pd.Series, 
                         timeframe: str) -> Dict[str, float]:
        """Analyze a specific timeframe using advanced algorithms and momentum"""
        # Resample data to the specified timeframe
        if timeframe != '1d':
            days = self.timeframes[timeframe]
            prices = prices.resample(f'{days}D').last()
            volumes = volumes.resample(f'{days}D').sum()
        
        # Calculate momentum indicators with divergence
        momentum_data = self.advanced_algo.calculate_momentum_indicators(
            prices=prices,
            volumes=volumes
        )
        
        # Extract individual indicators
        rocm = momentum_data['rocm']
        mfi = momentum_data['mfi']
        stoch_rsi = momentum_data['stoch_rsi']
        price_divergence = momentum_data['price_divergence']
        divergence_strength = momentum_data['strength_divergence']
        
        # Get trend and reversal characteristics
        trend_strength = momentum_data['trend_strength']
        trend_consistency = momentum_data['trend_consistency']
        trend_acceleration = momentum_data['trend_acceleration']
        reversal_prob = momentum_data['reversal_probability']
        reversal_signals = momentum_data['reversal_signals']
        reversal_confidence = momentum_data['reversal_confidence']
        
        # Calculate all signals
        momentum_signal = self._calculate_momentum_signal(rocm, mfi, stoch_rsi)
        divergence_signal = self._calculate_divergence_signal(
            price_divergence, divergence_strength, timeframe
        )
        trend_signal = self._calculate_trend_signal(
            trend_strength, trend_consistency, trend_acceleration
        )
        
        # Combine all signals
        final_momentum = self._combine_all_signals(
            momentum_signal, divergence_signal, trend_signal, timeframe
        )
        
        # Adjust weights based on momentum strength
        weights = self._get_momentum_adjusted_weights(timeframe, final_momentum)
        """
        Analyze a specific timeframe using advanced algorithms
        """
        # Resample data to the specified timeframe
        if timeframe != '1d':
            days = self.timeframes[timeframe]
            prices = prices.resample(f'{days}D').last()
            volumes = volumes.resample(f'{days}D').sum()
        
        # 1. Get MACD signals
        macd_signals, _, _, _, strength = self.advanced_algo.calculate_advanced_macd(
            prices=prices,
            fast=12,
            slow=26,
            signal=9,
            threshold=0.2
        )
        
        # 2. Calculate RSI and get divergence signals
        rsi = self._calculate_rsi(prices)
        rsi_signals = self.advanced_algo.detect_rsi_divergence(
            prices=prices,
            rsi_values=rsi,
            window=14
        )
        
        # 3. Get support/resistance levels
        sr_levels, sr_strengths = self.advanced_algo.calculate_ml_support_resistance(
            prices=prices,
            n_levels=5
        )
        
        # 4. Get Bollinger Band signals
        bb_signals, upper, lower, volatility = self.advanced_algo.calculate_dynamic_bollinger(
            prices=prices,
            window=20
        )
        
        # 5. Get VWAP signals
        vwap_signals, vwap, _ = self.advanced_algo.calculate_vwap_signals(
            prices=prices,
            volumes=volumes,
            window=20
        )
        
        # Combine all signals with custom weights for each timeframe
        weights = self._get_timeframe_weights(timeframe)
        combined_signal = self.advanced_algo.combine_signals(
            macd_signals,
            rsi_signals,
            bb_signals,
            vwap_signals,
            weights=weights
        )
        
        # Calculate proximity to support/resistance
        current_price = prices.iloc[-1]
        sr_proximity = self._calculate_sr_proximity(current_price, sr_levels, sr_strengths)
        
        # Add momentum analysis to results
        momentum_analysis = {
            'rocm': float(rocm.iloc[-1]),
            'mfi': float(mfi.iloc[-1]),
            'stoch_rsi': float(stoch_rsi.iloc[-1]),
            'momentum_signal': float(momentum_signal),
            'divergence_signal': float(divergence_signal),
            'final_momentum': float(final_momentum),
            'divergence_strength': float(divergence_strength.iloc[-1]),
            'trend_strength': float(trend_strength.iloc[-1]),
            'trend_consistency': float(trend_consistency.iloc[-1]),
            'trend_acceleration': float(trend_acceleration.iloc[-1]),
            'reversal_probability': float(reversal_prob.iloc[-1]),
            'reversal_signals': int(reversal_signals.iloc[-1]),
            'reversal_confidence': float(reversal_confidence.iloc[-1])
        }
        
        return {
            'signal_strength': float(combined_signal.iloc[-1]),
            'volatility': float(volatility.iloc[-1]),
            'sr_proximity': sr_proximity,
            'support_level': float(min(sr_levels)),
            'resistance_level': float(max(sr_levels)),
            'momentum': momentum_analysis
        }
    
    def get_multi_timeframe_analysis(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, Dict]:
        """
        Analyze all timeframes and return comprehensive results
        """
        results = {}
        for timeframe in self.timeframes.keys():
            results[timeframe] = self.analyze_timeframe(prices, volumes, timeframe)
        
        # Calculate overall signal incorporating all timeframes
        signals = [data['signal_strength'] * self._get_timeframe_importance(tf) 
                  for tf, data in results.items()]
        overall_signal = sum(signals)
        
        results['overall'] = {
            'signal_strength': overall_signal,
            'recommendation': self._get_recommendation(overall_signal),
            'confidence': self._calculate_confidence(results)
        }
        
        return results
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _get_momentum_adjusted_weights(self, timeframe: str, momentum_signal: float) -> List[float]:
        """Get indicator weights adjusted by momentum strength"""
        base_weights = self._get_timeframe_weights(timeframe)
        
        # Adjust weights based on momentum strength
        momentum_strength = abs(momentum_signal)
        if momentum_strength > 0.7:  # Strong momentum
            # Increase RSI and MACD weights
            base_weights = [
                base_weights[0] * 1.2,  # MACD
                base_weights[1] * 1.15, # RSI
                base_weights[2] * 0.8,  # BB
                base_weights[3] * 0.85  # VWAP
            ]
        elif momentum_strength < 0.3:  # Weak momentum
            # Increase BB and VWAP weights
            base_weights = [
                base_weights[0] * 0.9,  # MACD
                base_weights[1] * 0.85, # RSI
                base_weights[2] * 1.15, # BB
                base_weights[3] * 1.1   # VWAP
            ]
        
        # Normalize weights to sum to 1
        total = sum(base_weights)
        return [w/total for w in base_weights]
    
    def _calculate_divergence_signal(self, price_div: pd.Series, 
                                   strength: pd.Series, timeframe: str) -> float:
        """Calculate divergence signal with timeframe-specific weights"""
        # Get recent divergences (last N periods based on timeframe)
        # Optimized lookback periods based on timeframe volatility
        lookback = {
            '1d': 5,   # 5 days for more stable daily signals
            '2w': 3,   # 3 periods (6 weeks) for medium-term
            '1m': 2,   # 2 months for monthly
            '2m': 2,   # 4 months for bi-monthly
            '3m': 1    # 3 months for quarterly
        }[timeframe]
        
        # Timeframe-specific strength multipliers
        strength_mult = {
            '1d': 0.85,  # Reduce daily noise
            '2w': 0.95,  # Slightly reduced 2-week
            '1m': 1.0,   # Full strength monthly
            '2m': 1.0,   # Full strength bi-monthly
            '3m': 1.1    # Enhanced quarterly signals
        }[timeframe]
        
        # Get recent divergences
        recent_div = price_div.iloc[-lookback:]
        recent_strength = strength.iloc[-lookback:]
        
        # Weight more recent divergences higher
        weights = np.exp(np.linspace(0, 1, lookback))
        weights = weights / weights.sum()
        
        # Enhanced divergence signal calculation
        if len(recent_div) > 0:
            # Calculate base signals
            div_signal = np.average(recent_div, weights=weights)
            strength_avg = np.average(recent_strength, weights=weights)
            
            # Apply timeframe-specific strength multiplier
            weighted_signal = div_signal * strength_avg * strength_mult
            
            # Enhance strong signals, dampen weak ones
            if abs(weighted_signal) > 0.7:
                weighted_signal *= 1.2  # Boost strong signals
            elif abs(weighted_signal) < 0.3:
                weighted_signal *= 0.8  # Dampen weak signals
            
            return np.clip(weighted_signal, -1, 1)
        return 0.0
    
    def _calculate_trend_signal(self, strength: pd.Series, consistency: pd.Series, 
                              acceleration: pd.Series) -> float:
        """Calculate combined trend signal"""
        # Get latest values
        latest_strength = strength.iloc[-1]
        latest_consistency = consistency.iloc[-1]
        latest_acceleration = acceleration.iloc[-1]
        
        # Base weights
        strength_weight = 0.5
        consistency_weight = 0.3
        acceleration_weight = 0.2
        
        # Adjust weights based on consistency
        if abs(latest_consistency) > 0.8:  # Very consistent trend
            strength_weight = 0.6
            consistency_weight = 0.3
            acceleration_weight = 0.1
        elif abs(latest_consistency) < 0.4:  # Inconsistent trend
            strength_weight = 0.4
            consistency_weight = 0.3
            acceleration_weight = 0.3  # More weight on acceleration
        
        # Calculate weighted trend signal
        trend_signal = (
            latest_strength * strength_weight +
            latest_consistency * consistency_weight +
            latest_acceleration * acceleration_weight
        )
        
        return np.clip(trend_signal, -1, 1)
    
    def _adjust_for_reversal(self, signal: float, reversal_prob: float, 
                             reversal_conf: float, timeframe: str) -> float:
        """Adjust signal based on reversal detection"""
        if reversal_conf < 0.4:  # Low confidence reversals ignored
            return signal
        
        # Scale factor based on timeframe (longer timeframes need stronger confirmation)
        scale = {
            '1d': 0.8,   # More responsive to reversals
            '2w': 0.7,
            '1m': 0.6,
            '2m': 0.5,
            '3m': 0.4    # Most conservative
        }[timeframe]
        
        # Calculate reversal impact
        reversal_impact = reversal_prob * reversal_conf * scale
        
        # Apply reversal adjustment (reverses signal direction)
        if abs(reversal_impact) > 0.5:  # Strong reversal signal
            # Flip signal direction but maintain some original signal influence
            adjusted = -signal * reversal_impact
        else:  # Weak to moderate reversal
            # Reduce signal strength
            adjusted = signal * (1 - reversal_impact)
        
        return np.clip(adjusted, -1, 1)
    
    def _combine_all_signals(self, momentum: float, divergence: float, 
                            trend: float, timeframe: str) -> float:
        """Combine momentum, divergence, and trend signals"""
        # Base weights adjusted by timeframe
        if timeframe in ['1d', '2w']:  # Shorter timeframes
            weights = {
                'momentum': 0.45,
                'divergence': 0.30,
                'trend': 0.25
            }
        elif timeframe == '1m':  # Monthly
            weights = {
                'momentum': 0.35,
                'divergence': 0.35,
                'trend': 0.30
            }
        else:  # Longer timeframes
            weights = {
                'momentum': 0.30,
                'divergence': 0.35,
                'trend': 0.35
            }
        
        # Adjust weights based on trend consistency
        if abs(trend) > 0.7:  # Strong trend
            weights['trend'] += 0.1
            weights['momentum'] -= 0.05
            weights['divergence'] -= 0.05
        
        # Get reversal data for final signal
        reversal_prob = float(self.momentum_data['reversal_probability'].iloc[-1])
        reversal_conf = float(self.momentum_data['reversal_confidence'].iloc[-1])
        
        # Calculate initial combined signal
        combined = (
            momentum * weights['momentum'] +
            divergence * weights['divergence'] +
            trend * weights['trend']
        )
        
        # Adjust signal for potential reversals
        combined = self._adjust_for_reversal(
            combined, reversal_prob, reversal_conf, timeframe
        )
        
        # Amplify signal on strong agreement
        signals = [momentum, divergence, trend]
        if all(s > 0 for s in signals) or all(s < 0 for s in signals):
            avg_strength = sum(abs(s) for s in signals) / 3
            if avg_strength > 0.7:
                combined *= 1.3
            elif avg_strength > 0.5:
                combined *= 1.2
            else:
                combined *= 1.1
        
        return np.clip(combined, -1, 1)
    
    def _combine_momentum_divergence(self, momentum: float, divergence: float) -> float:
        """Combine momentum and divergence signals"""
        # Dynamic weight calculation based on signal strengths
        base_momentum_weight = 0.65
        base_divergence_weight = 0.35
        
        # Adjust weights based on signal strengths
        mom_strength = abs(momentum)
        div_strength = abs(divergence)
        
        if div_strength > 0.7:  # Strong divergence
            if mom_strength < 0.3:  # Weak momentum
                momentum_weight = 0.35
                divergence_weight = 0.65
            else:  # Both strong
                momentum_weight = 0.55
                divergence_weight = 0.45
        elif mom_strength > 0.7:  # Strong momentum
            if div_strength < 0.3:  # Weak divergence
                momentum_weight = 0.75
                divergence_weight = 0.25
            else:  # Both moderate to strong
                momentum_weight = 0.65
                divergence_weight = 0.35
        else:  # Both moderate or weak
            momentum_weight = base_momentum_weight
            divergence_weight = base_divergence_weight
        
        # Combine signals
        combined = (momentum * momentum_weight + divergence * divergence_weight)
        
        # Enhanced signal agreement amplification
        if (momentum > 0 and divergence > 0) or (momentum < 0 and divergence < 0):
            agreement_strength = min(mom_strength, div_strength)
            if agreement_strength > 0.7:
                combined *= 1.3  # Strong agreement boost
            elif agreement_strength > 0.4:
                combined *= 1.2  # Moderate agreement boost
            else:
                combined *= 1.1  # Weak agreement boost
        
        return np.clip(combined, -1, 1)
    
    def _calculate_momentum_signal(self, rocm: pd.Series, mfi: pd.Series, 
                                 stoch_rsi: pd.Series) -> float:
        """Calculate combined momentum signal"""
        # Get latest values
        latest_rocm = rocm.iloc[-1]
        latest_mfi = mfi.iloc[-1]
        latest_stoch_rsi = stoch_rsi.iloc[-1]
        
        # Normalize ROCM to -1 to 1 range
        norm_rocm = np.clip(latest_rocm / 10, -1, 1)
        
        # Normalize MFI to -1 to 1 range
        norm_mfi = (latest_mfi - 50) / 50
        
        # Normalize Stochastic RSI to -1 to 1 range
        norm_stoch_rsi = (latest_stoch_rsi - 50) / 50
        
        # Weighted combination
        momentum_signal = (
            norm_rocm * 0.4 +      # Rate of Change
            norm_mfi * 0.35 +      # Money Flow Index
            norm_stoch_rsi * 0.25  # Stochastic RSI
        )
        
        return momentum_signal
    
    def _get_timeframe_weights(self, timeframe: str) -> List[float]:
        """Get optimal weights for each algorithm based on timeframe"""
        # Optimized indicator weights [MACD, RSI, BB, VWAP]
        weights = {
            # Daily: Quick signals with volume confirmation
            '1d':  [0.32, 0.23, 0.20, 0.25],  # Increased VWAP for volume validation
            
            # 2 weeks: Enhanced trend detection
            '2w':  [0.28, 0.32, 0.25, 0.15],  # Stronger RSI for trend confirmation
            
            # Monthly: Primary decision making
            '1m':  [0.20, 0.40, 0.30, 0.10],  # Maximum RSI weight for trend strength
            
            # Bi-monthly: Strong trend validation
            '2m':  [0.15, 0.45, 0.35, 0.05],  # Heavy emphasis on RSI and BB patterns
            
            # Quarterly: Major trend identification
            '3m':  [0.10, 0.50, 0.35, 0.05]   # Dominant RSI for long-term trends
        }
        return weights[timeframe]
    
    def _get_timeframe_importance(self, timeframe: str) -> float:
        """Get the importance weight of each timeframe"""
        # Enhanced timeframe weights for better monthly signals
        weights = {
            '1d': 0.12,  # Reduced daily noise
            '2w': 0.18,  # Short-term trend confirmation
            '1m': 0.35,  # Increased monthly importance
            '2m': 0.22,  # Strong bi-monthly validation
            '3m': 0.13   # Long-term trend context
        }
        return weights[timeframe]
    
    def _calculate_sr_proximity(self, price: float, levels: pd.Series, 
                              strengths: pd.Series) -> float:
        """Calculate proximity to support/resistance levels"""
        distances = np.abs(levels - price) / price
        weighted_proximity = np.sum(strengths * (1 - distances)) / np.sum(strengths)
        return float(weighted_proximity)
    
    def _get_recommendation(self, signal: float) -> str:
        """Convert signal strength to trading recommendation"""
        if signal > 0.7:
            return "STRONG BUY"
        elif signal > 0.3:
            return "BUY"
        elif signal < -0.7:
            return "STRONG SELL"
        elif signal < -0.3:
            return "SELL"
        return "HOLD"
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence level of the analysis with timeframe agreement"""
        # Extract signals excluding overall
        timeframe_signals = {tf: data['signal_strength'] 
                           for tf, data in results.items() if tf != 'overall'}
        
        # Calculate trend agreement between consecutive timeframes
        trend_agreement = 0
        timeframes = list(self.timeframes.keys())
        for i in range(len(timeframes) - 1):
            current_tf = timeframes[i]
            next_tf = timeframes[i + 1]
            if (timeframe_signals[current_tf] * timeframe_signals[next_tf] > 0):
                trend_agreement += 1
        
        # Calculate signal strength and consistency
        signals = list(timeframe_signals.values())
        avg_strength = np.mean(np.abs(signals))
        signal_consistency = 1 - np.std(signals) / (np.max(np.abs(signals)) + 1e-6)
        
        # Enhanced confidence weighting system
        trend_weight = 0.45       # Increased trend agreement importance
        strength_weight = 0.30    # Moderate signal strength impact
        consistency_weight = 0.25  # Maintained consistency importance
        
        # Apply timeframe-based adjustments
        monthly_signal = timeframe_signals['1m']
        if abs(monthly_signal) > 0.7:  # Strong monthly signal
            trend_weight += 0.05
            strength_weight += 0.05
            consistency_weight -= 0.10
        
        confidence = (
            (trend_agreement / (len(timeframes) - 1)) * trend_weight +
            avg_strength * strength_weight +
            signal_consistency * consistency_weight
        ) * 100
        """Calculate confidence level of the analysis"""
        # Consider timeframe agreement and signal strengths
        signals = [abs(data['signal_strength']) for tf, data in results.items() 
                  if tf != 'overall']
        agreement = np.std(signals)  # Lower standard deviation means better agreement
        strength = np.mean(signals)  # Higher mean means stronger signals
        
        confidence = (strength * 0.7 + (1 - agreement) * 0.3) * 100
        return min(max(confidence, 0), 100)  # Ensure between 0 and 100
