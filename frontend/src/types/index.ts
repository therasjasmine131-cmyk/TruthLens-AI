export type PredictionLabel = "REAL" | "FAKE" | "UNCERTAIN";

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

export interface AnalysisResult {
  prediction: PredictionLabel;
  confidence: number;
  confidence_level: string;
  confidence_bands?: Record<string, number>;
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

export interface HealthStatus {
  status: string;
  backend: { status: string };
  database: { status: string; healthy: boolean };
  model: { status: string; model?: string; error?: string };
  vectorizer: { status: string; ready: boolean };
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
