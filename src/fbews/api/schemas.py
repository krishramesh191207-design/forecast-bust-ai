"""Pydantic response/request schemas for the API (PART 30)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    data_mode: str
    models_loaded: bool
    cycles_available: int
    time: str


class Location(BaseModel):
    lat: float
    lon: float
    region: str | None = None


class PredictedError(BaseModel):
    precipitation: float = Field(..., description="mm/day")
    temperature: float = Field(..., description="K")
    wind: float = Field(..., description="m/s")
    pressure: float = Field(..., description="hPa")


class GridPointResponse(BaseModel):
    valid_time: str
    forecast_cycle: str
    lead_day: int
    location: Location
    confidence: int = Field(..., ge=0, le=100,
                            description="Derived operational indicator: 100 x (1 - bust probability)")
    confidence_band: str
    bust_probability: float = Field(..., ge=0, le=1)
    predicted_error: PredictedError
    variable_bust_probability: dict[str, float] = {}
    dominant_variable: str
    regime: str
    forecast: dict
    ensemble: dict
    volatility: dict
    analogue: dict
    explanations: list[str]
    explanation_detail: dict
    uncertainty_decomposition: dict
    series: dict
    verification: dict | None = None
    data_banner: str


class InferenceRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=360)
    lead: int = Field(3, ge=1, le=10)
    cycle: str | None = Field(None, description="YYYY-MM-DD; defaults to the latest cycle")


# ---------------------------------------------------------------------------
# FEATURE 1 - Forecast Drift / cycle-to-cycle change
# ---------------------------------------------------------------------------
class ComparisonGrid(BaseModel):
    lat: list[float]
    lon: list[float]
    shape: list[int]
    resolution_deg: float


class ForecastComparisonResponse(BaseModel):
    available: bool = Field(...,
        description="False when the two cycles cannot be compared on the same valid time "
                    "or the layer has no 'better/worse' direction.")
    reason: str | None = Field(None,
        description="no_previous_cycle | same_valid_time_out_of_range | metric_not_defined | "
                    "grids_not_aligned")
    message: str | None = Field(None, description="Human readable explanation for the UI")
    cycle: str
    previousCycle: str | None = None
    validTime: str | None = Field(None, description="YYYY-MM-DD day both forecasts verify on")
    lead: int
    previousLead: int | None = None
    gapDays: int | None = None
    availableLeads: list[int] = Field(default_factory=list,
        description="Lead days that CAN be compared against previousCycle")
    metric: str
    label: str = ""
    units: str = ""
    decimals: int = 3
    higherIsBetter: bool | None = None
    threshold: float | None = None
    grid: ComparisonGrid | None = None
    current: list[float | None] = Field(default_factory=list)
    previous: list[float | None] = Field(default_factory=list)
    change: list[float | None] = Field(default_factory=list,
        description="current - previous, rounded to `decimals`")
    status: list[int] = Field(default_factory=list,
        description="0 no_change, 1 improved, 2 deteriorated, 3 unavailable")
    statusLabels: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# FEATURE 2 - Forecast Stability + confidence trajectory
# ---------------------------------------------------------------------------
class TrajectoryPoint(BaseModel):
    lead: int
    value: float


class TrajectoryStep(BaseModel):
    fromLead: int
    toLead: int
    delta: float


class TrajectoryReadout(BaseModel):
    points: list[TrajectoryPoint]
    n: int
    slope: float | None = Field(None, description="Change of the series per lead day")
    classification: str
    label: str
    largestDecline: TrajectoryStep | None = None
    largestImprovement: TrajectoryStep | None = None
    latestChange: TrajectoryStep | None = None
    available: bool


class VolatilityReadout(BaseModel):
    index: float | None
    signal: str = Field(..., description="calm | moderate | elevated | unavailable")
    low: float
    elevated: float
    available: bool


class CycleChangeReadout(BaseModel):
    available: bool
    current: int | None = None
    previous: int | None = None
    change: int | None = None
    status: int
    label: str
    previousCycle: str | None = None
    previousLead: int | None = None
    validTime: str | None = None
    reason: str | None = None


class StabilityContributions(BaseModel):
    leadTrend: int = Field(..., description="-1 | 0 | +1")
    cycleRevision: int = Field(..., description="-1 | 0 | +1")
    volatility: int = Field(..., description="-1 | 0 | +1")
    total: int = Field(..., ge=-3, le=3)
    cycleRevisionAvailable: bool
    volatilityAvailable: bool


class StabilityReadout(BaseModel):
    classification: str
    label: str
    score: int | None = Field(None, ge=0, le=100,
        description="Deterministic restatement of the Forecast Volatility Index: "
                    "round(100 * clamp(1 - index/3, 0, 1))")
    volatilitySignal: str
    contributions: StabilityContributions
    insufficientData: bool
    rules: dict[str, float]


class ForecastStabilityResponse(BaseModel):
    cycle: str
    previousCycle: str | None = None
    location: Location
    leadDay: int
    metric: str = Field("confidence", description="confidence | bust_probability")
    trajectory: TrajectoryReadout
    volatility: VolatilityReadout
    cycleChange: CycleChangeReadout
    stability: StabilityReadout


# ---------------------------------------------------------------------------
# FORECAST INTELLIGENCE / OPERATIONS CENTER schemas
# ---------------------------------------------------------------------------
class EnsembleVariable(BaseModel):
    label: str
    unit: str
    control: float | None = None
    ens_mean: float | None = None
    ens_std: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    q90: float | None = None
    iqr: float | None = None
    extras: dict[str, float | None] = {}
    observed: float | None = None
    available: bool
    reason: str | None = None


class EnsembleResponse(BaseModel):
    available: bool
    reason: str | None = None
    message: str | None = None
    cycle: str
    lead: int
    valid_time: str | None = None
    location: Location
    variables: dict[str, EnsembleVariable]
    member_count: int | None = None
    members_available: bool = False
    members_unavailable_reason: str
    observed_available: bool = False
    observed_reason: str | None = None
    data_mode: str | None = None
    warning: str | None = None


# ---------------------------------------------------------------------------
# OPERATIONS CENTER
# ---------------------------------------------------------------------------
class SystemRegionImpact(BaseModel):
    region: str
    cells: int
    meanConfidence: float | None = None
    maxBustProbability: float | None = None
    severity: str
    label: str


class SystemRecord(BaseModel):
    id: str
    type: str
    label: str
    lat: float
    lon: float
    depthHpa: float | None = None
    influenceRadiusDeg: float
    regionsAffected: list[str] = []
    nearestRegion: str | None = None
    meanConfidence: float | None = None
    severity: str
    labelImpact: str
    regionDetail: list[SystemRegionImpact] = []
    detectedFrom: str


class RegimeSummaryRow(BaseModel):
    regime: str
    cells: int
    share: float


class SystemsResponse(BaseModel):
    available: bool = True
    reason: str | None = None
    message: str | None = None
    cycle: str
    lead: int
    validTime: str
    systems: list[SystemRecord] = []
    notDetected: list[str] = []
    notDetectedReason: str
    influenceRadiusDeg: float
    regimeSummary: list[RegimeSummaryRow] = []
    regimeSummaryAvailable: bool = True


class WatchlistGroupsResponse(BaseModel):
    cycle: str
    lead: int | None = None
    groups: dict[str, dict[str, list[dict]]]
    counts: dict[str, int]


class AlertRuleModel(BaseModel):
    id: str
    label: str
    metric: str
    operator: str
    threshold: float
    lead: int | None = None
    severity: str = "warning"
    enabled: bool = True
    source: str = "project"


class AlertRecord(BaseModel):
    id: str
    ruleId: str
    ruleLabel: str
    metric: str
    operator: str
    threshold: float
    condition: str
    severity: str
    source: str
    region: str | None = None
    regionName: str | None = None
    lead: int
    value: float
    direction: str
    status: str = "suppressed"
    statusLabel: str = "Covered"
    previousSeverity: str | None = None
    severityRank: int = 0
    suppressedBecause: str | None = None


class AlertGroup(BaseModel):
    id: str
    ruleId: str
    label: str
    metric: str
    condition: str
    lead: int
    severity: str
    regions: list[str] = []
    regionNames: list[str] = []
    count: int
    worstValue: float | None = None
    statuses: list[str] = []


class AlertCounts(BaseModel):
    new: int = 0
    ongoing: int = 0
    escalated: int = 0
    total: int = 0
    groups: int = 0
    resolved: int = 0


class AlertsResponse(BaseModel):
    available: bool = True
    cycle: str
    previousCycle: str | None = None
    rules: list[AlertRuleModel] = []
    alerts: list[AlertRecord] = []
    groups: list[AlertGroup] = []
    suppressed: list[AlertRecord] = []
    resolved: list[AlertRecord] = []
    counts: AlertCounts
    bySeverity: dict[str, int] = {}
    previousCycleCompared: bool = False
    note: str


class EvaluateAlertsRequest(BaseModel):
    cycle: str | None = None
    rules: list[AlertRuleModel] = []
    previousCycle: str | None = None


# ---------------------------------------------------------------------------
# PHASE 5 - Event history / performance
# ---------------------------------------------------------------------------
class ScoreCounts(BaseModel):
    hits: int
    misses: int
    falseAlarms: int
    correctNegatives: int
    total: int
    events: int
    warnings: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    falseAlarmRatio: float | None = None
    frequency: float | None = None


class LeadScorecard(ScoreCounts):
    lead: int


class EventTypeScore(ScoreCounts):
    eventType: str


class Scorecard(BaseModel):
    overall: ScoreCounts
    byLead: list[LeadScorecard] = []
    byEventType: list[EventTypeScore] = []
    rule: str


class EarliestWarningRecord(BaseModel):
    eventDate: str
    eventType: str | None = None
    lat: float
    lon: float
    depthHpa: float | None = None
    leadsEvaluated: list[int] = []
    warningLeads: list[int] = []
    earliestWarningLead: int | None = None
    hoursBeforeValid: int | None = None
    issued: bool
    latestLeadEvaluated: int
    region: str | None = None


class EarliestSummary(BaseModel):
    eventsEvaluated: int
    eventsWithWarning: int
    eventsWithoutWarning: int
    detectionRate: float | None = None
    medianEarliestLead: float | None = None
    meanEarliestLead: float | None = None
    bestLead: int | None = None
    worstLead: int | None = None
    byEventType: list[dict] = []


class WarningPerformanceResponse(BaseModel):
    available: bool = True
    reason: str | None = None
    message: str | None = None
    source: str
    rule: str
    cyclesSampled: list[str] = []
    coverage: dict
    scorecard: Scorecard
    earliest: dict
    caveat: str


class EventImpactRow(BaseModel):
    eventType: str
    events: int
    regions: list[str] = []
    regionCounts: dict[str, int] = {}
    meanDepthHpa: float | None = None
    meanRadiusDeg: float | None = None


class EventImpactResponse(BaseModel):
    available: bool = True
    source: str
    method: str
    rows: list[EventImpactRow] = []
    uncovered: int = 0


# ---------------------------------------------------------------------------
# PHASE 7 - Region profile / trust / model drift
# ---------------------------------------------------------------------------
class RegionProfilePoint(BaseModel):
    cycle: str
    confidence: int
    bustProbability: float
    expectedErrorPrecipMmDay: float | None = None
    volatilityIndex: float | None = None


class RegionProfileResponse(BaseModel):
    available: bool = True
    reason: str | None = None
    region: str
    regionName: str
    cycle: str
    lead: int
    snapshot: dict
    history: list[RegionProfilePoint] = []
    systems: list[dict] = []
    alerts: list[dict] = []
    note: str


class TrustResponse(BaseModel):
    available: bool = True
    provenance: list[dict] = []
    thresholds: list[dict] = []
    artefacts: list[dict] = []
    limitations: list[str] = []
    members: dict
    generatedAt: str


class DriftPoint(BaseModel):
    label: str
    confidence: float | None = None
    bustProbability: float | None = None
    maePrecipMmDay: float | None = None


class DriftResponse(BaseModel):
    available: bool = True
    reason: str | None = None
    lead: int
    baselineLabel: str
    recentLabel: str
    baseline: DriftPoint
    recent: DriftPoint
    delta: dict
    interpretation: str
    caveat: str


# ---------------------------------------------------------------------------
# PHASE 8 - Forecast brief + search
# ---------------------------------------------------------------------------
class BriefSection(BaseModel):
    heading: str
    lines: list[str] = []


class ForecastBriefResponse(BaseModel):
    available: bool = True
    reason: str | None = None
    cycle: str
    lead: int
    region: str | None = None
    headline: str
    sections: list[BriefSection] = []
    caveats: list[str] = []
    generatedAt: str


class SearchHit(BaseModel):
    type: str
    label: str
    detail: str | None = None
    view: str | None = None
    tab: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = []
    namespaces: list[str] = []
