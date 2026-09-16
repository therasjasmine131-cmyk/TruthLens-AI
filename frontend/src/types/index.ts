export type PredictionLabel = "REAL" | "FAKE" | "UNCERTAIN";
export type Verdict = "REAL" | "FALSE" | "UNVERIFIED";
export type FinalVerdict = "TRUE" | "FALSE" | "UNVERIFIED";

export interface Probabilities {
  real: number;
  fake: number;
  uncertain: number;
}

export interface Keyword {
  term: string;
  score: number;
}

export interface ArticleStats {
  word_count: number;
  character_count: number;
  sentence_count: number;
  average_sentence_length: number;
  unique_words: number;
  vocabulary_richness: number;
  capitalized_words: number;
  exclamation_marks: number;
  question_marks: number;
}

export interface ExplanationFeature {
  term: string;
  weight: number;
  contribution: number;
  influence: "positive" | "negative" | "low";
}

export interface Explanation {
  method: string;
  model_class: string;
  features: ExplanationFeature[];
  note: string;
  direction_label: string;
}

export interface ModelInfo {
  name: string;
  model_class?: string;
  vectorizer?: string;
  dataset_source?: string;
  train_samples?: number;
  test_samples?: number;
  n_features?: number;
  training_date?: string;
  explainability?: string;
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    roc_auc?: number;
  };
}

export interface LiveCheck {
  available: boolean;
  source?: string;
  label: "REAL" | "FAKE" | "UNVERIFIED";
  confidence: number;
  reasoning: string;
}

export interface AiVerdict {
  verdict: "REAL" | "FAKE" | "UNVERIFIED";
  confidence: number;
  reasoning?: string;
  source?: string;
}

export interface VerificationEvidence {
  claim: string;
  claim_verdict: Verdict;
  evidence_title: string;
  source: string | null;
  domain: string | null;
  url: string | null;
  date: string | null;
  relation: "SUPPORTS" | "CONTRADICTS" | "NEUTRAL";
  relevance: number;
  source_quality: number;
  source_tier: "primary" | "secondary" | "tertiary" | "unknown";
  source_reasons: string[];
  retrieved_from: string;
}

export type AiStageDecision = "SUPPORT" | "CONTRADICT" | "INSUFFICIENT";

export interface AiAnalysis1 {
  available: boolean;
  source: string;
  model?: string;
  decision: AiStageDecision;
  confidence: number;
  reasoning: string;
}

export interface AiReview {
  available: boolean;
  source: string;
  model?: string;
  verdict: AiStageDecision;
  confidence: number;
  agrees_with_first: boolean;
  problems: string[];
  reasoning: string;
}

export interface ClaimStages {
  NN_RESULT?: { prediction: string | null; confidence: number | null };
  EVIDENCE_RESULT?: {
    verdict: Verdict;
    confidence: number;
    supporting_count: number;
    contradicting_count: number;
    independent_sources: boolean;
  };
  AI_RESULT_1?: AiAnalysis1 | null;
  AI_REVIEW_RESULT?: AiReview | null;
  FINAL_RESULT?: { verdict: Verdict; confidence: number; authority: string };
}

export interface VerificationClaim {
  text: string;
  type: string;
  opinion?: boolean;
  prediction?: boolean;
  negation?: boolean;
  verdict: Verdict;
  confidence: number;
  confidence_label: string;
  reason: string;
  evidence: Record<string, unknown>[];
  cross_source: {
    distinct_support_domains: number;
    distinct_contradict_domains: number;
    independent_support: boolean;
    independent_contradiction: boolean;
  };
  ml?: { prediction: string; confidence: number } | null;
  ml_prediction?: string | null;
  ml_confidence?: number | null;
  ai_analysis_1?: AiAnalysis1 | null;
  ai_review?: AiReview | null;
  conflicts?: string[];
  final_verdict?: Verdict;
  final_confidence?: number;
  final_authority?: string;
  supporting_counts?: number;
  contradicting_counts?: number;
  source_credibility?: {
    items: number;
    distinct_domains: number;
    avg_source_quality: number;
    tiers: Record<string, number>;
    independent: boolean;
  };
  stages?: ClaimStages | null;
}

export interface VerificationOverall {
  verdict: Verdict;
  confidence: number;
  confidence_label: string;
  mixed: boolean;
  counts: { real: number; false: number; unverified: number; total_claims: number };
  explanation: string;
}

export interface Verification {
  status: string;
  language: { code: string; label: string };
  claims: VerificationClaim[];
  overall: VerificationOverall;
  evidence_matrix: VerificationEvidence[];
  pipeline: {
    language_detected: string;
    claims_extracted: number;
    evidence_items: number;
    sources_used: string[];
    live_evidence_used: boolean;
    ai_used?: boolean;
    ai_claims_analyzed?: number;
    ai_reviews_completed?: number;
  };
  stages?: {
    PIPELINE: string[];
    claims_analyzed: number;
    ai_available: boolean;
    ai_claims: number;
    agreement: Record<string, unknown>;
  } | null;
  ml_article?: { prediction: string; confidence: number; probabilities: { real: number; fake: number } } | null;
}

export interface AnalysisResult {
  prediction: PredictionLabel;
  confidence: number;
  confidence_level: string;
  confidence_bands?: Record<string, number>;
  decided?: boolean;
  uncertain_threshold?: number;
  probabilities: Probabilities;
  model_raw?: { p_real: number; p_fake: number };
  model: string;
  model_info: ModelInfo;
  keywords: Keyword[];
  article_stats: ArticleStats;
  explanation: Explanation;
  disclaimer: string;
  saved: boolean;
  history_id?: number;
  headline_only?: boolean;
  caveat?: string | null;
  live_check?: LiveCheck | null;
  ai_verdict?: AiVerdict | null;
  verdict?: Verdict | null;
  verification?: Verification | null;
}

export interface HistoryItem {
  id: number;
  headline: string;
  prediction: PredictionLabel;
  real_probability: number;
  fake_probability: number;
  uncertain_probability: number;
  confidence: number;
  model_name: string;
  word_count: number;
  character_count: number;
  sentence_count: number;
  top_keywords: string[];
  created_at: string;
}

export interface HistoryDetail extends HistoryItem {
  article_text: string;
  analysis_metadata: Record<string, unknown>;
}

export interface HistoryPage {
  items: HistoryItem[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
  has_prev: boolean;
  has_next: boolean;
}

export type VerifyLanguageMode = "auto" | "english" | "tamil" | "tanglish";

export interface VerifyResponse {
  status: string;
  final_verdict: FinalVerdict;
  confidence: number;
  reasoning: string;
  language_mode: VerifyLanguageMode;
  language_detected?: string;
  claims_analyzed: number;
  evidence_items: number;
  sources_used: string[];
  evidence_matrix: VerificationEvidence[];
  claims: VerificationClaim[];
  live_check: LiveCheck | null;
  verification: Verification | null;
  stages?: UnknownRecord | null;
  notes?: { verdict_basis?: string };
}

export interface UnknownRecord {
  [key: string]: unknown;
}

export interface HealthStatus {
  status: string;
  backend: { status: string };
  database: { status: string; healthy: boolean };
  model: { status: string; model?: string; error?: string };
  neural_network: { status: string; ready: boolean; architecture?: string };
  model_name?: string;
}

export interface AnalyticsData {
  total_analyses: number;
  label_counts: Record<PredictionLabel, number>;
  average_confidence: number;
  model_counts: Record<string, number>;
  confidence_histogram: { bucket: string; count: number }[];
  trend: {
    bucket: string;
    real: number;
    fake: number;
    uncertain: number;
    total: number;
  }[];
  bucket: string;
  days: number;
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number | null;
  val_f1?: number;
  support?: number;
  confusion_matrix?: number[][];
  roc_curve?: { fpr: number[]; tpr: number[] } | null;
}

export interface ModelPerformanceData {
  trained: boolean;
  message?: string;
  best_model: string | null;
  best_metrics?: ModelMetrics;
  best_confusion_matrix?: number[][];
  models: ({ name: string } & ModelMetrics)[];
  dataset_source?: string;
  train_samples?: number;
  test_samples?: number;
  n_features?: number;
  training_date?: string;
  class_labels?: string[];
}

export interface DatasetStats {
  source: string;
  total_records: number;
  real_count: number;
  fake_count: number;
  class_balance: { real: number; fake: number };
  missing_values: number;
  duplicate_count: number;
  average_article_length: number;
  length_histogram: { bucket: string; count: number }[];
}

export interface DatasetStatsResponse {
  stats: DatasetStats;
  using_raw: boolean;
}

export interface DatasetSampleRow {
  headline: string;
  text_preview: string;
  label: string;
  subject: string;
}

export interface DemoArticle {
  id: number;
  label: string;
  headline: string;
  article: string;
  kind: string;
  disclaimer: string;
}

export interface BatchResult {
  total: number;
  completed: number;
  errors: number;
  real: number;
  fake: number;
  uncertain: number;
  results: BatchRowResult[];
}

export interface BatchRowResult {
  index: number;
  headline?: string;
  prediction?: PredictionLabel;
  confidence?: number;
  probabilities?: Probabilities;
  error?: string;
}

export interface SettingsData {
  theme: "light" | "dark" | "system";
  max_article_length: number;
  max_headline_length: number;
  confidence_levels: {
    very_high_min: number;
    high_min: number;
    moderate_min: number;
  };
}

export type AiTextLabel = "Likely AI" | "Likely Human" | "Uncertain";

export interface AiTextResult {
  ai_generated_score: number;
  label: AiTextLabel;
  backend: "heuristic" | "gemini" | "roberta";
  signals: Record<string, number>;
  input_chars: number;
  gemini_score?: number;
  transformer_score?: number;
  warning?: string;
  error?: string;
  status?: string;
}

export interface NewsArticle {
  id: number;
  source: string;
  author: string;
  headline: string;
  description: string;
  article: string;
  url: string;
  image_url: string;
  published_at: string;
}

export interface TrendingNewsResponse {
  country: string;
  total: number;
  items: NewsArticle[];
}
