import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface VitalChartProps {
  data: any[];
  valueKey: string;
  entropyKey: string;
  label: string;
  unit: string;
  icon: React.ReactNode;
  thresholdLow?: number;
  thresholdHigh?: number;
  drugEvents?: any[];
  isContributing?: boolean;
  severityColor?: string;
  currentValue?: number | null;
  currentTrend?: string;
  domainMin?: number;
  domainMax?: number;
}

const TrendIcon = ({ trend }: { trend?: string }) => {
  const size = 14;
  if (trend === 'rising') return <TrendingUp size={size} className="text-[#F46B52]" />;
  if (trend === 'falling') return <TrendingDown size={size} className="text-[#E9A52F]" />;
  return <Minus size={size} className="text-[#52677D]" />;
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || payload.length === 0) return null;
  
  const time = label ? new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  const vitalPayload = payload.find((p: any) => p.dataKey && !p.dataKey.includes('entropy'));
  const entropyPayload = payload.find((p: any) => p.dataKey && p.dataKey.includes('entropy'));

  return (
    <div className="bg-white border border-[#DCE4E7] shadow-lg rounded-lg p-3 text-xs font-sans">
      <div className="font-bold text-[#102A43] mb-2 border-b border-[#DCE4E7] pb-1">{time}</div>
      {vitalPayload && (
        <div className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full bg-[#4FA8B8]" />
          <span className="font-bold text-[#102A43]">{vitalPayload.value?.toFixed(1)}</span>
        </div>
      )}
      {entropyPayload && entropyPayload.value != null && (
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#9b59b6]" />
          <span className="font-bold text-[#102A43]">{entropyPayload.value.toFixed(3)} <span className="text-[#52677D] font-normal">entropy</span></span>
        </div>
      )}
    </div>
  );
};

export const VitalChart: React.FC<VitalChartProps> = ({
  data,
  valueKey,
  entropyKey,
  label,
  unit,
  icon,
  thresholdLow,
  thresholdHigh,
  drugEvents = [],
  isContributing = false,
  severityColor = '#4FA8B8',
  currentValue,
  currentTrend,
  domainMin,
  domainMax,
}) => {
  const valueDomain = useMemo(() => {
    if (domainMin != null && domainMax != null) return [domainMin, domainMax];
    if (!data || data.length === 0) return ['auto', 'auto'];

    const values = data.map((d) => d[valueKey]).filter((v) => v != null);
    if (values.length === 0) return ['auto', 'auto'];

    let min = Math.min(...values);
    let max = Math.max(...values);

    if (thresholdLow != null) min = Math.min(min, thresholdLow);
    if (thresholdHigh != null) max = Math.max(max, thresholdHigh);

    const padding = (max - min) * 0.1 || 5;
    return [
      domainMin != null ? domainMin : Math.floor(min - padding),
      domainMax != null ? domainMax : Math.ceil(max + padding),
    ];
  }, [data, valueKey, thresholdLow, thresholdHigh, domainMin, domainMax]);

  const formatXTick = (timestamp: any) => {
    if (!timestamp) return '';
    const d = new Date(timestamp);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  const xTickInterval = useMemo(() => {
    if (!data) return 0;
    if (data.length < 30) return 0;
    if (data.length < 120) return Math.floor(data.length / 6);
    return Math.floor(data.length / 8);
  }, [data]);

  return (
    <div 
      className={`bg-white rounded-xl border border-[#DCE4E7] overflow-hidden flex flex-col ${isContributing ? 'ring-2 ring-opacity-50' : ''}`} 
      style={isContributing ? { borderColor: severityColor, boxShadow: `0 0 0 2px ${severityColor}80` } : {}}
    >
      {/* Header */}
      <div className="bg-[#F7F7F4] px-4 py-2 border-b border-[#DCE4E7] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[#52677D]">{icon}</span>
          <span className="font-bold text-[#102A43] text-sm tracking-wide">{label}</span>
          {isContributing && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold text-white uppercase ml-2" style={{ backgroundColor: severityColor }}>
              Contributing
            </span>
          )}
        </div>
        <div className="flex items-end gap-2">
          <div className="text-xl font-bold text-[#102A43]">
            {currentValue != null ? currentValue.toFixed(1) : '--'}
          </div>
          <div className="text-xs text-[#52677D] mb-1">{unit}</div>
          <div className="mb-1"><TrendIcon trend={currentTrend} /></div>
        </div>
      </div>

      {/* Chart Body */}
      <div className="h-32 w-full p-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 0, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id={`grad-${valueKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4FA8B8" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#4FA8B8" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#DCE4E7" vertical={false} />

            <XAxis
              dataKey="time"
              tickFormatter={formatXTick}
              tick={{ fontSize: 10, fill: '#52677D', fontFamily: 'monospace' }}
              axisLine={{ stroke: '#DCE4E7' }}
              tickLine={false}
              interval={xTickInterval}
            />

            <YAxis
              yAxisId="value"
              domain={valueDomain}
              tick={{ fontSize: 10, fill: '#52677D', fontFamily: 'monospace' }}
              axisLine={false}
              tickLine={false}
              width={40}
            />

            <YAxis
              yAxisId="entropy"
              orientation="right"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#9b59b6', fontFamily: 'monospace' }}
              axisLine={false}
              tickLine={false}
              width={30}
              tickCount={3}
            />

            <Tooltip content={<CustomTooltip />} />

            {thresholdLow != null && (
              <ReferenceLine yAxisId="value" y={thresholdLow} stroke="#F46B52" strokeDasharray="3 3" strokeWidth={1} />
            )}
            {thresholdHigh != null && (
              <ReferenceLine yAxisId="value" y={thresholdHigh} stroke="#F46B52" strokeDasharray="3 3" strokeWidth={1} />
            )}

            {drugEvents.map((de, i) => (
              <ReferenceLine
                key={`drug-${i}`}
                yAxisId="value"
                x={de.time}
                stroke="#E9A52F"
                strokeDasharray="5 3"
                strokeWidth={1}
                label={{ value: de.drugName, position: 'top', fontSize: 9, fill: '#E9A52F' }}
              />
            ))}

            <Area
              yAxisId="value"
              type="monotone"
              dataKey={valueKey}
              stroke="none"
              fill={`url(#grad-${valueKey})`}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              yAxisId="value"
              type="monotone"
              dataKey={valueKey}
              stroke="#4FA8B8"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />

            <Line
              yAxisId="entropy"
              type="monotone"
              dataKey={entropyKey}
              stroke="#9b59b6"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
