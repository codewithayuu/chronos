import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

interface CESGaugeProps {
  value: number | null | undefined;
  rawValue?: number | null;
  severityLabel: string;
  color: string;
}

export const CESGauge: React.FC<CESGaugeProps> = ({ value, rawValue, severityLabel, color }) => {
  const percentage = ((value ?? 0) * 100).toFixed(1);

  // Build the arc for the gauge
  const gaugeArc = useMemo(() => {
    const radius = 72;
    const centerX = 90;
    const centerY = 82;
    const startAngle = -210;
    const endAngle = 30;
    const totalAngle = endAngle - startAngle;
    const fillAngle = startAngle + totalAngle * Math.min(value ?? 0, 1);

    const toRad = (deg: number) => (deg * Math.PI) / 180;

    const bgStartX = centerX + radius * Math.cos(toRad(startAngle));
    const bgStartY = centerY + radius * Math.sin(toRad(startAngle));
    const bgEndX = centerX + radius * Math.cos(toRad(endAngle));
    const bgEndY = centerY + radius * Math.sin(toRad(endAngle));

    const fillEndX = centerX + radius * Math.cos(toRad(fillAngle));
    const fillEndY = centerY + radius * Math.sin(toRad(fillAngle));

    const largeArcBg = totalAngle > 180 ? 1 : 0;
    const fillArc = fillAngle - startAngle;
    const largeArcFill = fillArc > 180 ? 1 : 0;

    const bgPath = `M ${bgStartX} ${bgStartY} A ${radius} ${radius} 0 ${largeArcBg} 1 ${bgEndX} ${bgEndY}`;
    const fillPath = `M ${bgStartX} ${bgStartY} A ${radius} ${radius} 0 ${largeArcFill} 1 ${fillEndX} ${fillEndY}`;

    return { bgPath, fillPath, fillEndX, fillEndY };
  }, [value]);

  const showDrugAdjustment = rawValue != null && value != null && Math.abs(rawValue - value) > 0.001;

  return (
    <div className="flex flex-col items-center justify-center p-6 bg-white rounded-2xl border border-[#DCE4E7] shadow-2xs relative overflow-hidden">
      
      {/* Background Glow */}
      <div 
        className="absolute inset-0 opacity-5"
        style={{ background: `radial-gradient(circle at center, ${color} 0%, transparent 70%)` }}
      />

      <div className="relative w-48 h-32 flex items-center justify-center mb-4">
        <svg viewBox="0 0 180 120" className="w-full h-full drop-shadow-sm">
          <defs>
            <linearGradient id="ces-gauge-grad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={color} stopOpacity={0.4} />
              <stop offset="100%" stopColor={color} stopOpacity={1} />
            </linearGradient>
            <filter id="ces-glow">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background arc */}
          <path
            d={gaugeArc.bgPath}
            fill="none"
            stroke="#F7F7F4"
            strokeWidth={10}
            strokeLinecap="round"
          />
          <path
            d={gaugeArc.bgPath}
            fill="none"
            stroke="#DCE4E7"
            strokeWidth={1}
            strokeLinecap="round"
          />

          {/* Filled arc */}
          <motion.path
            d={gaugeArc.fillPath}
            fill="none"
            stroke="url(#ces-gauge-grad)"
            strokeWidth={10}
            strokeLinecap="round"
            filter="url(#ces-glow)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
          />

          {/* End dot */}
          <motion.circle
            cx={gaugeArc.fillEndX}
            cy={gaugeArc.fillEndY}
            r={5}
            fill="white"
            stroke={color}
            strokeWidth={2}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.6 }}
          />
        </svg>

        {/* Central value */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-8">
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="text-4xl font-black tracking-tight"
            style={{ color: color }}
          >
            {value != null ? value.toFixed(2) : '--'}
          </motion.div>
          <span className="text-[10px] font-bold text-[#52677D] tracking-widest uppercase">Composite Entropy</span>
        </div>
      </div>

      {/* Sub-metrics */}
      <div className="w-full flex items-center justify-between px-4 pt-4 border-t border-[#DCE4E7]">
        {showDrugAdjustment ? (
          <div className="flex items-center gap-3 w-full justify-center">
            <div className="flex flex-col items-center">
              <span className="text-[10px] text-[#52677D] font-bold uppercase">Raw</span>
              <span className="text-sm font-mono font-bold text-[#102A43]">{rawValue?.toFixed(2)}</span>
            </div>
            <span className="text-[#DCE4E7]">→</span>
            <div className="flex flex-col items-center">
              <span className="text-[10px] text-[#52677D] font-bold uppercase">Adjusted</span>
              <span className="text-sm font-mono font-bold" style={{ color: color }}>{value?.toFixed(2)}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center w-1/2 border-r border-[#DCE4E7]">
            <span className="text-[10px] text-[#52677D] font-bold uppercase tracking-wider">Score</span>
            <span className="text-sm font-mono font-bold text-[#102A43]">{percentage}%</span>
          </div>
        )}

        {!showDrugAdjustment && (
          <div className="flex flex-col items-center w-1/2">
            <span className="text-[10px] text-[#52677D] font-bold uppercase tracking-wider">Status</span>
            <span className="text-sm font-bold uppercase" style={{ color: color }}>
              {severityLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
