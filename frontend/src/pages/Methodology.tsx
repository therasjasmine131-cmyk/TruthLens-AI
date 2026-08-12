import {
  BookOpen,
  Braces,
  Target,
  BarChart3,
  Grid3x3,
  AlertTriangle,
  GraduationCap,
  Scale,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";

const CARDS: { title: string; icon: typeof BookOpen; body: string }[] = [
  {
    title: "What is fake news detection?",
    icon: AlertTriangle,
    body: "Fake news detection is the task of automatically classifying a piece of text as real or fake. It combines natural-language processing (to understand the text) with supervised machine learning (to learn patterns from labelled examples).",
  },
  {
    title: "What is NLP?",
    icon: Braces,
    body: "Natural-language processing is the field of AI that lets computers read, understand and derive meaning from human language. Here we use NLP to clean articles, split sentences, count words and convert text into numeric features a classifier can use.",
  },
  {
    title: "What is TF-IDF?",
    icon: Target,
    body: "Term Frequency–Inverse Document Frequency converts text into numbers. Each word gets a score: how often it appears in this article (term frequency) weighted against how rare it is across the whole training corpus (inverse document frequency). Rare, distinctive words therefore get high scores.",
  },
  {
    title: "What is supervised learning?",
    icon: GraduationCap,
    body: "Supervised learning trains a model on labelled examples: pairs of text and its known label (REAL or FAKE). The model learns a mapping from text to label, then generalises to new, unseen text.",
  },
  {
    title: "How does Logistic Regression work?",
    icon: BarChart3,
    body: "Logistic regression learns a weighted sum of the TF-IDF features and passes it through the sigmoid function to produce a probability between 0 and 1. Positive weights push toward one class, negative weights toward the other. The weights are exactly what we use for explainability.",
  },
  {
    title: "Why compare multiple models?",
    icon: Scale,
    body: "No single model is universally best. Different algorithms make different assumptions; comparing them on the same held-out test set shows which generalises best to this dataset. We select the best model by validation F1 score.",
  },
];

const METRICS: { term: string; definition: string }[] = [
  { term: "Accuracy", definition: "Fraction of all predictions that are correct. Misleading when classes are imbalanced." },
  { term: "Precision", definition: "Of everything predicted REAL, how much was actually REAL. High precision = few false alarms." },
  { term: "Recall", definition: "Of all actually-REAL articles, how many were found. High recall = few misses." },
  { term: "F1 Score", definition: "Harmonic mean of precision and recall. A balanced single number for model quality." },
  { term: "ROC-AUC", definition: "Area under the ROC curve: probability that a random REAL article scores higher than a random FAKE one. 0.5 = random, 1.0 = perfect." },
  { term: "Confusion Matrix", definition: "A table of actual vs predicted classes. Cells count true positives, true negatives, false positives and false negatives." },
  { term: "False Positive", definition: "A FAKE article predicted as REAL (a false alarm that makes a fake story look credible)." },
  { term: "False Negative", definition: "A REAL article predicted as FAKE (a real story wrongly dismissed)." },
];

const LIMITS: string[] = [
  "The model learned from 2016-2017 US political news (ISOT dataset). Language and events drift, so accuracy on current or foreign news is lower.",
  "Dataset bias: real examples come mostly from Reuters; fake examples from partisan sources. The model may over-rely on style signals (e.g. 'reuters', exclamation marks) rather than truth.",
  "AI confidence is not proof. The model outputs probabilities based on patterns; it cannot verify facts or check sources.",
  "The three-way output (REAL/FAKE/UNCERTAIN) is derived from the binary model's decision margin. Near-50/50 predictions are reported as UNCERTAIN.",
  "Short texts (single sentences) carry too little signal and are less reliable.",
  "Adversarial or carefully-crafted fake articles designed to mimic real reporting can fool the classifier.",
];

export function Methodology() {
  return (
    <div>
      <PageHeader
        title="Methodology"
        subtitle="A plain-language guide to the machine-learning concepts behind TruthLens AI — ideal for presentations and vivas."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {CARDS.map(({ title, icon: Icon, body }) => (
          <Card key={title} className="p-5">
            <div className="mb-3 flex items-center gap-2">
              <div className="rounded-lg bg-primary-50 p-2 text-primary-600 dark:bg-primary-950 dark:text-primary-400">
                <Icon size={16} />
              </div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">{body}</p>
          </Card>
        ))}
      </div>

      <Card className="mt-6 overflow-hidden">
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
            <Grid3x3 size={15} className="text-primary-500" /> Key evaluation metrics
          </h3>
        </div>
        <div className="grid gap-px bg-slate-100 dark:bg-slate-800 sm:grid-cols-2">
          {METRICS.map(({ term, definition }) => (
            <div key={term} className="bg-white px-5 py-4 dark:bg-[#111a2e]">
              <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{term}</p>
              <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">{definition}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card className="mt-6">
        <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
            <BookOpen size={15} className="text-primary-500" /> Model limitations and dataset bias
          </h3>
        </div>
        <ul className="list-disc space-y-2 px-8 py-5 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          {LIMITS.map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
        <div className="border-t border-slate-200 px-5 py-4 dark:border-slate-800">
          <p className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-800 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
            TruthLens AI provides machine-learning-based estimates from patterns learned from its
            training data. A prediction is not proof that an article is true or false. Always verify
            important claims using reliable sources.
          </p>
        </div>
      </Card>
    </div>
  );
}
