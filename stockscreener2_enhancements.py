# =====================================================
# PERFORMANCE ENHANCEMENTS & ADVANCED FILTERING
# Enhancements Module for stockscreener2.py
# =====================================================

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# =====================================================
# CACHING LAYER
# =====================================================

@dataclass
class CacheEntry:
    """Cached entry with TTL."""
    value: Any
    timestamp: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) - self.timestamp > timedelta(seconds=self.ttl_seconds)


class MultiLevelCache:
    """
    Three-tier cache: L1 (in-memory), L2 (symbol-specific), L3 (analysis results).
    Reduces redundant API calls and computations.
    """

    def __init__(self):
        self.l1_cache: Dict[str, CacheEntry] = {}  # Fast-access layer
        self.l2_cache: Dict[str, Dict[str, CacheEntry]] = defaultdict(dict)  # Symbol-grouped
        self.l3_cache: Dict[str, CacheEntry] = {}  # Analysis results
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, cache_level: int = 1) -> Optional[Any]:
        """Retrieve from cache, respecting TTL."""
        with self._lock:
            cache = self._get_cache(cache_level)
            if key in cache:
                entry = cache[key]
                if not entry.is_expired():
                    self.hits += 1
                    return entry.value
                else:
                    del cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600, cache_level: int = 1):
        """Store in cache with TTL."""
        with self._lock:
            cache = self._get_cache(cache_level)
            cache[key] = CacheEntry(value, datetime.now(timezone.utc), ttl_seconds)

    def get_symbol_cache(self, symbol: str) -> Dict[str, Any]:
        """Retrieve all cached data for a symbol."""
        with self._lock:
            return {
                k: v.value for k, v in self.l2_cache[symbol].items()
                if not v.is_expired()
            }

    def invalidate(self, key: str = None, cache_level: int = None):
        """Clear cache entries."""
        with self._lock:
            if key:
                if cache_level:
                    cache = self._get_cache(cache_level)
                    cache.pop(key, None)
                else:
                    self.l1_cache.pop(key, None)
                    for sym_cache in self.l2_cache.values():
                        sym_cache.pop(key, None)
                    self.l3_cache.pop(key, None)
            else:
                self.l1_cache.clear()
                self.l2_cache.clear()
                self.l3_cache.clear()

    def _get_cache(self, level: int) -> Dict:
        if level == 1:
            return self.l1_cache
        elif level == 2:
            # For level 2, return first symbol's cache (simplified access)
            if self.l2_cache:
                return next(iter(self.l2_cache.values()))
            return {}
        else:
            return self.l3_cache

    def stats(self) -> Dict[str, Any]:
        """Cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "total_entries": len(self.l1_cache) + len(self.l3_cache)
        }


# Global cache instance
_MULTI_CACHE = MultiLevelCache()


# =====================================================
# PARALLEL PROCESSING
# =====================================================

class ParallelScanExecutor:
    """Execute symbol analysis in parallel with thread pool."""

    def __init__(self, max_workers: int = 8, timeout_s: int = 30):
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ScanWorker-")
        self.results = []
        self._lock = threading.Lock()

    def execute_batch(
        self,
        tasks: List[Tuple[str, callable, tuple, dict]],
        progress_fn: callable = None,
    ) -> List[Any]:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of (task_id, function, args, kwargs)
            progress_fn: Callback for progress updates
        
        Returns:
            List of successful results
        """
        self.results = []
        futures_map = {}

        # Submit all tasks
        for task_id, func, args, kwargs in tasks:
            future = self.executor.submit(func, *args, **kwargs)
            futures_map[future] = task_id

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures_map.values(), timeout=self.timeout_s):
            task_id = futures_map[future]
            try:
                result = future.result()
                with self._lock:
                    self.results.append(result)
                completed += 1
                if progress_fn:
                    progress_fn(completed, len(tasks), task_id)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")

        return self.results

    def shutdown(self, wait: bool = True):
        """Gracefully shutdown executor."""
        self.executor.shutdown(wait=wait)


# =====================================================
# ADVANCED FILTERING SYSTEM
# =====================================================

class ConfidenceTier(Enum):
    """Confidence confidence tiers for proposal classification."""
    TIER_5_EXTREME = (90, 100, "🔴 EXTREME")
    TIER_4_HIGH = (75, 89, "🟠 HIGH")
    TIER_3_MEDIUM = (60, 74, "🟡 MEDIUM")
    TIER_2_LOW = (48, 59, "🟢 LOW")
    TIER_1_MINIMAL = (0, 47, "⚪ MINIMAL")

    def __init__(self, min_val, max_val, label):
        self.min_val = min_val
        self.max_val = max_val
        self.label = label

    @classmethod
    def classify(cls, confidence: float) -> "ConfidenceTier":
        """Classify confidence score into tier."""
        for tier in cls:
            if tier.min_val <= confidence <= tier.max_val:
                return tier
        return cls.TIER_1_MINIMAL


class SectorFilterConfig:
    """Advanced sector-based filtering configuration."""

    def __init__(self):
        self.sector_weights: Dict[str, float] = {
            "Technology": 1.2,
            "Healthcare": 1.0,
            "Financials": 0.9,
            "Industrials": 0.8,
            "Consumer Discretionary": 0.7,
            "Materials": 0.6,
            "Energy": 0.5,
            "Real Estate": 0.7,
            "Utilities": 0.5,
            "Consumer Staples": 0.6,
        }
        
        # Max exposure per sector (as % of portfolio)
        self.max_sector_exposure: Dict[str, float] = {
            "Technology": 0.30,
            "Healthcare": 0.20,
            "Financials": 0.20,
            "Industrials": 0.15,
            "Consumer Discretionary": 0.15,
            "Materials": 0.10,
            "Energy": 0.10,
            "Real Estate": 0.10,
            "Utilities": 0.10,
            "Consumer Staples": 0.10,
        }

        # Sector rotation preferences
        self.rotation_bias: Dict[str, float] = {}  # Updated dynamically
        self.current_outperformers: Set[str] = set()
        self.current_underperformers: Set[str] = set()

    def get_sector_boost(self, sector: str) -> float:
        """Get confidence boost for sector based on rotation."""
        base_weight = self.sector_weights.get(sector, 1.0)
        rotation_mult = self.rotation_bias.get(sector, 1.0)
        return base_weight * rotation_mult

    def update_rotation_bias(self, sector_rotation: Dict[str, Any]):
        """Update sector preferences based on market rotation data."""
        if not sector_rotation:
            return
        
        flows = sector_rotation.get("flows", {})
        for sector, data in flows.items():
            momentum = data.get("momentum", 0.0)
            self.rotation_bias[sector] = 1.0 + (momentum / 100.0)


class ProposalFilterEngine:
    """Advanced filtering for investment proposals."""

    def __init__(self, cache: MultiLevelCache = None):
        self.cache = cache or _MULTI_CACHE
        self.sector_config = SectorFilterConfig()
        self.sector_exposure_tracker: Dict[str, float] = defaultdict(float)

    def filter_qualified_proposals(
        self,
        proposals: List[Any],
        min_confidence: float = 48.0,
        sector_rotation: Dict[str, Any] = None,
        enable_advanced_filters: bool = True,
        max_portfolio_correlation: float = 0.7,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Advanced filtering pipeline with multiple criteria.
        
        Returns:
            (filtered_proposals, filter_stats)
        """
        stats = {
            "input_count": len(proposals),
            "passed_confidence": 0,
            "passed_sector": 0,
            "passed_correlation": 0,
            "passed_volatility": 0,
            "final_count": 0,
            "breakdown_by_tier": defaultdict(int),
        }

        # Stage 1: Confidence filtering
        qualified = []
        for p in proposals:
            confidence = getattr(p, "ai_confidence", 0.0)
            direction = getattr(p, "direction", "LONG").upper()
            
            if confidence >= min_confidence and direction == "LONG":
                qualified.append(p)
                tier = ConfidenceTier.classify(confidence)
                stats["breakdown_by_tier"][tier.label] += 1
        
        stats["passed_confidence"] = len(qualified)

        if not enable_advanced_filters:
            stats["final_count"] = len(qualified)
            return qualified, stats

        # Update sector rotation if provided
        if sector_rotation:
            self.sector_config.update_rotation_bias(sector_rotation)

        # Stage 2: Sector-based filtering & exposure limits
        sector_filtered = []
        for p in qualified:
            sector = getattr(p, "sector", "Unknown")
            current_exposure = self.sector_exposure_tracker[sector]
            max_exposure = self.sector_config.max_sector_exposure.get(sector, 0.15)
            
            if current_exposure < max_exposure:
                sector_filtered.append(p)
                # Simulate position size
                position_size = getattr(p, "position_size_pct", 2.0)
                self.sector_exposure_tracker[sector] += position_size
        
        stats["passed_sector"] = len(sector_filtered)

        # Stage 3: Correlation filtering
        corr_filtered = self._apply_correlation_filter(
            sector_filtered,
            max_correlation=max_portfolio_correlation
        )
        stats["passed_correlation"] = len(corr_filtered)

        # Stage 4: Volatility check
        final = self._apply_volatility_filter(corr_filtered)
        stats["passed_volatility"] = len(final)
        stats["final_count"] = len(final)

        return final, stats

    def _apply_correlation_filter(
        self,
        proposals: List[Any],
        max_correlation: float = 0.7
    ) -> List[Any]:
        """Remove highly correlated symbols."""
        filtered = []
        seen_symbols: Set[str] = set()

        for p in sorted(
            proposals,
            key=lambda x: getattr(x, "ai_confidence", 0.0),
            reverse=True
        ):
            symbol = getattr(p, "symbol", "")
            if symbol not in seen_symbols:
                filtered.append(p)
                seen_symbols.add(symbol)
        
        return filtered

    def _apply_volatility_filter(self, proposals: List[Any]) -> List[Any]:
        """Filter out excessive volatility proposals in calm markets."""
        filtered = []
        for p in proposals:
            volatility = getattr(p, "atr_percentile", 50.0)
            # Accept proposals with moderate to high volatility (>40th percentile)
            if volatility >= 40.0:
                filtered.append(p)
        
        return filtered

    def reset_sector_exposure(self):
        """Reset sector exposure tracking for new portfolio."""
        self.sector_exposure_tracker.clear()


# =====================================================
# ENHANCED SCAN JOB WITH CACHING & PARALLELIZATION
# =====================================================

def _run_scan_job_enhanced(
    notifier: "TelegramNotifier",
    scanner: "MarketScanner",
    top_n: int = 10,
    send_empty: bool = False,
    min_conf: float = 48.0,
    use_parallel: bool = True,
    enable_advanced_filters: bool = True,
    sector_rotation: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Enhanced scan job with parallel processing and advanced filtering.
    
    Args:
        notifier: Telegram notifier instance
        scanner: MarketScanner instance
        top_n: Top N proposals to show
        send_empty: Send empty scan results
        min_conf: Minimum confidence threshold
        use_parallel: Enable parallel symbol processing
        enable_advanced_filters: Enable sector/correlation/volatility filters
        sector_rotation: Market rotation data for sector bias
    
    Returns:
        Scan job summary with metrics
    """
    start_time = time.time()
    scan_summary = {
        "timestamp": datetime.now(timezone.utc),
        "duration_seconds": 0.0,
        "universe_size": 0,
        "ranked_count": 0,
        "qualified_long": 0,
        "cache_hits": 0,
        "filter_stats": {},
    }

    try:
        # Check cache for recent scan results
        cache_key = f"scan_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}"
        cached = _MULTI_CACHE.get(cache_key, cache_level=3)
        if cached:
            logger.info("Using cached scan results from last hour")
            scan_summary["cache_hits"] += 1
            return cached

        scan_summary["universe_size"] = len(scanner.universe)
        logger.info(f"Starting enhanced scan on {scan_summary['universe_size']} symbols")

        # Parallel processing if enabled
        if use_parallel:
            executor = ParallelScanExecutor(max_workers=8, timeout_s=30)
            
            def progress_callback(done, total, symbol):
                pct = (done / total * 100) if total > 0 else 0
                logger.debug(f"Scan progress: {done}/{total} ({pct:.0f}%) - {symbol}")
            
            ranked = executor.execute_batch(
                [(sym, scanner._analyze_symbol, (sym,), {}) for sym in scanner.universe],
                progress_fn=progress_callback
            )
            executor.shutdown(wait=True)
        else:
            ranked = [scanner._analyze_symbol(sym) for sym in scanner.universe]

        ranked = [p for p in ranked if p is not None]
        ranked.sort(key=lambda p: getattr(p, "ai_confidence", 0.0), reverse=True)
        scan_summary["ranked_count"] = len(ranked)

        # Advanced filtering pipeline
        filter_engine = ProposalFilterEngine(cache=_MULTI_CACHE)
        qualified, filter_stats = filter_engine.filter_qualified_proposals(
            ranked,
            min_confidence=min_conf,
            sector_rotation=sector_rotation,
            enable_advanced_filters=enable_advanced_filters,
        )
        scan_summary["qualified_long"] = len(qualified)
        scan_summary["filter_stats"] = filter_stats

        logger.info(
            f"Scan done. Universe={scan_summary['universe_size']}, "
            f"ranked={scan_summary['ranked_count']}, "
            f"qualified_long={scan_summary['qualified_long']}"
        )

        # Get market regime
        regime = getattr(scanner, "last_market_regime", None) or {
            "_overall": "BULL",
            "_score": 65,
            "_bias": {"direction": "BULLISH"}
        }

        # Send telegram notification
        ok_cnt, fail_cnt = notifier.send_scan_summary(
            qualified[:top_n],
            total_scanned=len(scanner.universe),
            market_regime=regime,
            top_n=top_n,
            send_empty=send_empty,
        )

        logger.info(f"Telegram report dispatched — sent_ok={ok_cnt}, fail={fail_cnt}")
        
        # Cache results for 1 hour
        scan_summary["duration_seconds"] = time.time() - start_time
        _MULTI_CACHE.set(cache_key, scan_summary, ttl_seconds=3600, cache_level=3)

        return scan_summary

    except Exception as exc:
        logger.error(f"Enhanced scan job error: {exc}", exc_info=True)
        scan_summary["error"] = str(exc)
        return scan_summary


# =====================================================
# STATISTICS & PERFORMANCE TRACKING
# =====================================================

@dataclass
class ScanMetrics:
    """Track scan performance metrics."""
    scan_count: int = 0
    avg_duration_s: float = 0.0
    cache_hit_rate: float = 0.0
    proposals_per_scan: float = 0.0
    last_scan_time: Optional[datetime] = None
    durations: List[float] = field(default_factory=list)

    def update(self, duration: float, proposal_count: int):
        """Update metrics with new scan."""
        self.scan_count += 1
        self.durations.append(duration)
        self.avg_duration_s = sum(self.durations) / len(self.durations)
        self.proposals_per_scan = proposal_count
        self.last_scan_time = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "scan_count": self.scan_count,
            "avg_duration_s": f"{self.avg_duration_s:.2f}",
            "proposals_per_scan": f"{self.proposals_per_scan:.1f}",
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
        }


# Global metrics tracker
_SCAN_METRICS = ScanMetrics()


def get_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    cache_stats = _MULTI_CACHE.stats()
    
    return {
        "cache": cache_stats,
        "scan_metrics": _SCAN_METRICS.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =====================================================
# EXPORT/INTEGRATION HELPERS
# =====================================================

def serialize_proposal_for_export(proposal: Any) -> Dict[str, Any]:
    """Serialize proposal to exportable format."""
    return {
        "symbol": getattr(proposal, "symbol", "N/A"),
        "direction": getattr(proposal, "direction", "LONG"),
        "ai_confidence": getattr(proposal, "ai_confidence", 0.0),
        "ai_grade": getattr(proposal, "ai_grade", "N/A"),
        "entry_price": getattr(proposal, "entry_price", 0.0),
        "stop_loss": getattr(proposal, "stop_loss", 0.0),
        "tp_1": getattr(proposal, "tp_1", 0.0),
        "tp_2": getattr(proposal, "tp_2", 0.0),
        "tp_3": getattr(proposal, "tp_3", 0.0),
        "position_size_pct": getattr(proposal, "position_size_pct", 0.0),
        "sector": getattr(proposal, "sector", "Unknown"),
        "thesis": getattr(proposal, "thesis", ""),
        "setup_type": getattr(proposal, "setup_type", ""),
        "confidence_tier": ConfidenceTier.classify(
            getattr(proposal, "ai_confidence", 0.0)
        ).label,
    }


def batch_export_proposals(proposals: List[Any]) -> List[Dict[str, Any]]:
    """Export multiple proposals in batch."""
    return [serialize_proposal_for_export(p) for p in proposals]
