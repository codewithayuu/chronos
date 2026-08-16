import React, { useEffect, useRef } from 'react';
import katex from 'katex';

interface MathFormulaProps {
  math: string;
  displayMode?: boolean;
  className?: string;
}

export const MathFormula: React.FC<MathFormulaProps> = ({
  math,
  displayMode = true,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(math, containerRef.current, {
          displayMode,
          throwOnError: false,
          strict: false,
        });
      } catch (err) {
        console.warn('KaTeX rendering error:', err);
        if (containerRef.current) {
          containerRef.current.innerText = math;
        }
      }
    }
  }, [math, displayMode]);

  return (
    <div
      ref={containerRef}
      className={`katex-rendered overflow-x-auto text-[#102A43] ${className}`}
      aria-label={`Mathematical formula: ${math}`}
    />
  );
};
