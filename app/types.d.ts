declare module '@/components/ModelSelector' {
  interface ModelSelectorProps {
    selectedModel: string;
    onModelSelect: (model: string) => void;
  }
  const ModelSelector: React.FC<ModelSelectorProps>;
  export default ModelSelector;
}

declare module '@/components/FeedbackBox' {
  interface FeedbackBoxProps {
    feedback: string;
  }
  const FeedbackBox: React.FC<FeedbackBoxProps>;
  export default FeedbackBox;
}

declare module '@/components/LookupBox' {
  interface LookupBoxProps {
    className?: string;
  }
  const LookupBox: React.FC<LookupBoxProps>;
  export default LookupBox;
}

declare module '@/utils/processFrame' {
  export function processFrame(
    videoElement: HTMLVideoElement | null,
    canvasElement: HTMLCanvasElement | null,
    selectedModel: string
  ): Promise<string>;
} 