export interface EvaluationResult {
  success: boolean;
  letter?: string | null;
  confidence?: number | null;
  feedback?: string | null;
  error?: string | null;
}

export type SignEvaluationHandler = (
  imageData: string,
  expectedSign: string,
  result: EvaluationResult
) => Promise<void>; 