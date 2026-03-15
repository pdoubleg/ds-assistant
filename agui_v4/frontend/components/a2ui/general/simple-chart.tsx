"use client";

/**
 * SimpleChart Component
 *
 * Renders simple bar, line, or pie charts using pure SVG.
 * No external charting library required.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Check, Copy, Maximize2, Minimize2 } from "lucide-react";

export interface SimpleChartProps {
  chart_type: "bar" | "line" | "pie";
  title: string;
  labels: string[];
  values: number[];
}

const CATEGORICAL_CHART_COLORS = [
  "#06748C",
  "#1A1446",
  "#FFD000",
  "#28A3AF",
  "#343741",
  "#29254F",
  "#78E1E1",
  "#FFDB50",
  "#99E5EA",
  "#565656",
  "#FFE280",
  "#AEEDED",
];

const LINE_CHART_COLORS = ["#06748C"];

/**
 * Returns the built-in palette for the chart type.
 *
 * Bar and pie charts use an extended categorical palette derived from the
 * Liberty brand colors. Line charts intentionally stay on a single accent
 * color so trend data remains visually consistent across the app.
 */
function getDefaultChartColors(chartType: SimpleChartProps["chart_type"]): string[] {
  return chartType === "line" ? LINE_CHART_COLORS : CATEGORICAL_CHART_COLORS;
}

/**
 * Formats chart numeric values for readability.
 *
 * Large numbers are abbreviated with K/M/B suffixes for cleaner axis labels.
 *
 * Example:
 *   formatChartValue(1500)   => "1.5K"
 *   formatChartValue(42)     => "42"
 */
function formatChartValue(value: number): string {
  if (Math.abs(value) >= 1_000_000_000)
    return (value / 1_000_000_000).toFixed(1).replace(/\.0$/, "") + "B";
  if (Math.abs(value) >= 1_000_000)
    return (value / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (Math.abs(value) >= 1_000)
    return (value / 1_000).toFixed(1).replace(/\.0$/, "") + "K";
  return new Intl.NumberFormat("en-US").format(value);
}

/** Full-precision format used for hover tooltips. */
function formatChartValueFull(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

/**
 * Computes "nice" gridline tick values for a given maximum.
 *
 * Returns an array of evenly-spaced round numbers from 0 up to (and
 * potentially slightly beyond) `maxVal`, producing readable axis labels.
 */
function computeGridTicks(maxVal: number, count = 5): number[] {
  if (maxVal <= 0) return [0];

  const rawStep = maxVal / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const residual = rawStep / magnitude;

  let niceStep: number;
  if (residual <= 1.5) niceStep = magnitude;
  else if (residual <= 3) niceStep = 2 * magnitude;
  else if (residual <= 7) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;

  const ticks: number[] = [];
  for (let v = 0; v <= maxVal + niceStep * 0.01; v += niceStep) {
    ticks.push(Math.round(v * 1e6) / 1e6);
  }

  // Guarantee the last tick covers maxVal so bars/lines never overflow
  if (ticks.length === 0 || ticks[ticks.length - 1] < maxVal) {
    const next = ticks.length > 0 ? ticks[ticks.length - 1] + niceStep : niceStep;
    ticks.push(Math.round(next * 1e6) / 1e6);
  }

  return ticks;
}

/**
 * Converts an SVG element to a PNG blob.
 */
async function svgToPngBlob(svgElement: SVGSVGElement): Promise<Blob> {
  const serializer = new XMLSerializer();
  const svgContent = serializer.serializeToString(svgElement);
  const svgWithNamespace = svgContent.includes("xmlns=")
    ? svgContent
    : svgContent.replace(
        "<svg",
        '<svg xmlns="http://www.w3.org/2000/svg"',
      );

  const svgBlob = new Blob([svgWithNamespace], {
    type: "image/svg+xml;charset=utf-8",
  });
  const objectUrl = URL.createObjectURL(svgBlob);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Unable to load chart image"));
      img.src = objectUrl;
    });

    const viewBox = svgElement.viewBox.baseVal;
    const width =
      (viewBox && viewBox.width > 0 ? viewBox.width : svgElement.clientWidth) ||
      600;
    const height =
      (viewBox && viewBox.height > 0 ? viewBox.height : svgElement.clientHeight) ||
      300;

    const devicePixelRatio = window.devicePixelRatio || 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * devicePixelRatio);
    canvas.height = Math.round(height * devicePixelRatio);
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      throw new Error("Unable to render chart image");
    }

    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.drawImage(image, 0, 0, width, height);

    const pngBlob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );

    if (!pngBlob) {
      throw new Error("Unable to encode chart image");
    }

    return pngBlob;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function BarChart({
  labels,
  values,
  colors,
  isExpanded = false,
}: {
  labels: string[];
  values: number[];
  colors: string[];
  isExpanded?: boolean;
}) {
  const maxVal = Math.max(...values, 1);
  const barWidth = isExpanded ? 52 : 40;
  const gap = isExpanded ? 20 : 16;
  const chartHeight = isExpanded ? 240 : 160;

  // Top padding keeps value labels above the tallest bar inside the viewBox
  const topPad = isExpanded ? 24 : 18;
  const bottomPad = isExpanded ? 28 : 22;
  const yAxisWidth = isExpanded ? 48 : 38;
  const svgWidth = yAxisWidth + labels.length * (barWidth + gap) + gap;
  const svgHeight = topPad + chartHeight + bottomPad;

  const ticks = computeGridTicks(maxVal);
  const niceMax = ticks[ticks.length - 1] || maxVal;

  const truncateLabel = (label: string, limit: number) => {
    if (label.length <= limit) return label;
    return label.slice(0, limit - 1) + "\u2026";
  };

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
    >
      <defs>
        {colors.map((color, i) => (
          <linearGradient
            key={`bar-grad-${i}`}
            id={`bar-grad-${i}`}
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop offset="0%" stopColor={color} stopOpacity={0.95} />
            <stop offset="100%" stopColor={color} stopOpacity={0.7} />
          </linearGradient>
        ))}
      </defs>

      {/* Horizontal gridlines and y-axis tick labels */}
      {ticks.map((tick) => {
        const y = topPad + chartHeight - (tick / niceMax) * chartHeight;
        return (
          <g key={`grid-${tick}`}>
            <line
              x1={yAxisWidth}
              y1={y}
              x2={svgWidth}
              y2={y}
              stroke="currentColor"
              strokeOpacity={0.08}
              strokeDasharray="4 3"
            />
            <text
              x={yAxisWidth - 6}
              y={y + 3}
              textAnchor="end"
              fontSize={isExpanded ? "10" : "8"}
              className="fill-current opacity-40"
              style={{ fill: "currentColor" }}
            >
              {formatChartValue(tick)}
            </text>
          </g>
        );
      })}

      {/* Bars */}
      {values.map((val, i) => {
        const barHeight = (val / niceMax) * chartHeight;
        const x = yAxisWidth + gap + i * (barWidth + gap);
        const y = topPad + chartHeight - barHeight;
        const colorIdx = i % colors.length;
        const total = values.reduce((s, v) => s + v, 0) || 1;
        const pct = ((val / total) * 100).toFixed(1);
        const barTip = `${labels[i]}\nValue: ${formatChartValueFull(val)}\n${pct}% of total`;

        return (
          <g key={i} className="group">
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx={isExpanded ? 6 : 4}
              fill={`url(#bar-grad-${colorIdx})`}
              className="transition-all duration-300 drop-shadow-sm"
              style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.08))" }}
            >
              <title>{barTip}</title>
            </rect>
            {/* Highlight on hover */}
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx={isExpanded ? 6 : 4}
              fill="white"
              opacity={0}
              className="transition-opacity duration-200 hover:opacity-[0.12] pointer-events-none"
            />
            {/* Value label above bar */}
            <text
              x={x + barWidth / 2}
              y={y - (isExpanded ? 8 : 5)}
              textAnchor="middle"
              fontSize={isExpanded ? "11" : "9"}
              fontWeight="600"
              className="fill-current opacity-60 pointer-events-none"
              style={{ fill: "currentColor" }}
            >
              {formatChartValue(val)}
            </text>
            {/* X-axis label */}
            <text
              x={x + barWidth / 2}
              y={topPad + chartHeight + (isExpanded ? 18 : 14)}
              textAnchor="middle"
              fontSize={isExpanded ? "10" : "8"}
              fontWeight="500"
              className="fill-current opacity-50 cursor-help"
              style={{ fill: "currentColor" }}
            >
              <title>{labels[i]}</title>
              {truncateLabel(labels[i], isExpanded ? 8 : 6)}
            </text>
          </g>
        );
      })}

      {/* Base axis line */}
      <line
        x1={yAxisWidth}
        y1={topPad + chartHeight}
        x2={svgWidth}
        y2={topPad + chartHeight}
        stroke="currentColor"
        strokeOpacity={0.12}
      />
    </svg>
  );
}

/**
 * Builds a smooth monotone cubic spline path through a set of points.
 *
 * The algorithm uses Fritsch-Carlson monotone interpolation so the curve
 * never overshoots data points — important for accurate chart rendering.
 */
function buildSmoothPath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return pts.map((p) => `M ${p.x} ${p.y}`).join(" ");
  if (pts.length === 2)
    return `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y}`;

  const n = pts.length;
  const dx: number[] = [];
  const dy: number[] = [];
  const slopes: number[] = [];

  for (let i = 0; i < n - 1; i++) {
    dx.push(pts[i + 1].x - pts[i].x);
    dy.push(pts[i + 1].y - pts[i].y);
    slopes.push(dy[i] / (dx[i] || 1));
  }

  // Tangents via monotone method
  const tangents: number[] = [slopes[0]];
  for (let i = 1; i < n - 1; i++) {
    if (slopes[i - 1] * slopes[i] <= 0) {
      tangents.push(0);
    } else {
      tangents.push((slopes[i - 1] + slopes[i]) / 2);
    }
  }
  tangents.push(slopes[n - 2]);

  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < n - 1; i++) {
    const segLen = dx[i] / 3;
    const cp1x = pts[i].x + segLen;
    const cp1y = pts[i].y + tangents[i] * segLen;
    const cp2x = pts[i + 1].x - segLen;
    const cp2y = pts[i + 1].y - tangents[i + 1] * segLen;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${pts[i + 1].x} ${pts[i + 1].y}`;
  }
  return d;
}

function LineChart({
  labels,
  values,
  colors,
  isExpanded = false,
}: {
  labels: string[];
  values: number[];
  colors: string[];
  isExpanded?: boolean;
}) {
  const maxVal = Math.max(...values, 1);
  const chartHeight = isExpanded ? 240 : 160;

  // Top padding keeps value labels above the highest point inside the viewBox
  const topPad = isExpanded ? 28 : 22;
  const bottomPad = isExpanded ? 28 : 22;
  const yAxisWidth = isExpanded ? 48 : 38;
  const rightPad = isExpanded ? 16 : 10;
  const svgWidth = Math.max(
    yAxisWidth + labels.length * (isExpanded ? 72 : 60) + rightPad,
    300,
  );
  const plotWidth = svgWidth - yAxisWidth - rightPad;
  const stepX = plotWidth / Math.max(labels.length - 1, 1);
  const color = colors[0] || LINE_CHART_COLORS[0];
  const svgHeight = topPad + chartHeight + bottomPad;

  const ticks = computeGridTicks(maxVal);
  const niceMax = ticks[ticks.length - 1] || maxVal;

  const points = values.map((val, i) => ({
    x: yAxisWidth + i * stepX,
    y: topPad + chartHeight - (val / niceMax) * chartHeight,
  }));

  const smoothPath = buildSmoothPath(points);

  // Baseline y sits at the bottom of the plot area
  const baseY = topPad + chartHeight;

  // Closed path for the gradient area fill
  const areaPath =
    smoothPath +
    ` L ${points[points.length - 1].x} ${baseY}` +
    ` L ${points[0].x} ${baseY} Z`;

  const truncateLabel = (label: string, limit: number) => {
    if (label.length <= limit) return label;
    return label.slice(0, limit - 1) + "\u2026";
  };

  const gradientId = `line-area-grad-${color.replace("#", "")}`;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0.02} />
        </linearGradient>
      </defs>

      {/* Horizontal gridlines and y-axis labels */}
      {ticks.map((tick) => {
        const y = topPad + chartHeight - (tick / niceMax) * chartHeight;
        return (
          <g key={`grid-${tick}`}>
            <line
              x1={yAxisWidth}
              y1={y}
              x2={svgWidth - rightPad}
              y2={y}
              stroke="currentColor"
              strokeOpacity={0.08}
              strokeDasharray="4 3"
            />
            <text
              x={yAxisWidth - 6}
              y={y + 3}
              textAnchor="end"
              fontSize={isExpanded ? "10" : "8"}
              className="fill-current opacity-40"
              style={{ fill: "currentColor" }}
            >
              {formatChartValue(tick)}
            </text>
          </g>
        );
      })}

      {/* Gradient area fill */}
      <path d={areaPath} fill={`url(#${gradientId})`} />

      {/* Smooth line */}
      <path
        d={smoothPath}
        fill="none"
        stroke={color}
        strokeWidth={isExpanded ? 2.5 : 2}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 1px 2px ${color}40)` }}
      />

      {/* Data points with ring style */}
      {points.map((p, i) => {
        const prev = i > 0 ? values[i - 1] : null;
        const delta =
          prev !== null
            ? `\nChange: ${values[i] >= prev ? "+" : ""}${formatChartValueFull(values[i] - prev)}`
            : "";
        const pointTip = `${labels[i]}\nValue: ${formatChartValueFull(values[i])}${delta}`;

        return (
          <g key={i} className="group cursor-pointer">
            {/* Invisible hit area for easier hover targeting */}
            <circle
              cx={p.x}
              cy={p.y}
              r={isExpanded ? 14 : 10}
              fill="transparent"
            >
              <title>{pointTip}</title>
            </circle>
            {/* Outer ring (visible on hover via CSS) */}
            <circle
              cx={p.x}
              cy={p.y}
              r={isExpanded ? 8 : 6}
              fill={color}
              opacity={0}
              className="transition-opacity duration-200 group-hover:opacity-[0.15] pointer-events-none"
            />
            {/* White ring background */}
            <circle
              cx={p.x}
              cy={p.y}
              r={isExpanded ? 5 : 4}
              fill="white"
              stroke={color}
              strokeWidth={isExpanded ? 2.5 : 2}
              className="transition-all duration-200 pointer-events-none"
            />
            {/* Value label */}
            <text
              x={p.x}
              y={p.y - (isExpanded ? 14 : 10)}
              textAnchor="middle"
              fontSize={isExpanded ? "11" : "9"}
              fontWeight="600"
              className="fill-current opacity-60 pointer-events-none"
              style={{ fill: "currentColor" }}
            >
              {formatChartValue(values[i])}
            </text>
            {/* X-axis label */}
            <text
              x={p.x}
              y={baseY + (isExpanded ? 18 : 14)}
              textAnchor="middle"
              fontSize={isExpanded ? "10" : "8"}
              fontWeight="500"
              className="fill-current opacity-50 cursor-help"
              style={{ fill: "currentColor" }}
            >
              <title>{labels[i]}</title>
              {truncateLabel(labels[i], isExpanded ? 10 : 8)}
            </text>
          </g>
        );
      })}

      {/* Base axis line */}
      <line
        x1={yAxisWidth}
        y1={baseY}
        x2={svgWidth - rightPad}
        y2={baseY}
        stroke="currentColor"
        strokeOpacity={0.12}
      />
    </svg>
  );
}

function PieChart({
  labels,
  values,
  colors,
  isExpanded = false,
}: {
  labels: string[];
  values: number[];
  colors: string[];
  isExpanded?: boolean;
}) {
  const total = values.reduce((sum, v) => sum + v, 0) || 1;
  const size = isExpanded ? 260 : 190;
  const cx = size / 2;
  const cy = size / 2;
  const outerRadius = isExpanded ? 105 : 78;
  // Donut inner radius — ~55% of outer gives a clean modern look
  const innerRadius = isExpanded ? 60 : 44;
  // Small angular gap (in radians) between slices
  const sliceGap = 0.025;

  let cumulative = 0;
  const slices = values.map((val, i) => {
    const startAngle =
      (cumulative / total) * 2 * Math.PI - Math.PI / 2 + sliceGap / 2;
    cumulative += val;
    const endAngle =
      (cumulative / total) * 2 * Math.PI - Math.PI / 2 - sliceGap / 2;

    // Outer arc endpoints
    const ox1 = cx + outerRadius * Math.cos(startAngle);
    const oy1 = cy + outerRadius * Math.sin(startAngle);
    const ox2 = cx + outerRadius * Math.cos(endAngle);
    const oy2 = cy + outerRadius * Math.sin(endAngle);

    // Inner arc endpoints (reversed direction)
    const ix1 = cx + innerRadius * Math.cos(endAngle);
    const iy1 = cy + innerRadius * Math.sin(endAngle);
    const ix2 = cx + innerRadius * Math.cos(startAngle);
    const iy2 = cy + innerRadius * Math.sin(startAngle);

    const largeArc = val / total > 0.5 ? 1 : 0;

    const path =
      `M ${ox1} ${oy1}` +
      ` A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${ox2} ${oy2}` +
      ` L ${ix1} ${iy1}` +
      ` A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix2} ${iy2}` +
      ` Z`;

    return {
      path,
      color: colors[i % colors.length],
      label: labels[i],
      value: val,
      percentage: ((val / total) * 100).toFixed(1),
    };
  });

  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-center md:gap-8">
      <div className="relative shrink-0" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="overflow-visible"
        >
          <defs>
            {colors.map((color, i) => (
              <linearGradient
                key={`pie-grad-${i}`}
                id={`pie-grad-${i}`}
                x1="0"
                y1="0"
                x2="1"
                y2="1"
              >
                <stop offset="0%" stopColor={color} stopOpacity={1} />
                <stop offset="100%" stopColor={color} stopOpacity={0.75} />
              </linearGradient>
            ))}
            <filter id="pie-shadow">
              <feDropShadow dx="0" dy="1" stdDeviation="2" floodOpacity="0.1" />
            </filter>
          </defs>
          {slices.map((slice, i) => {
            const sliceTip = `${slice.label}\nValue: ${formatChartValueFull(slice.value)}\n${slice.percentage}% of total`;
            return (
              <path
                key={i}
                d={slice.path}
                fill={`url(#pie-grad-${i % colors.length})`}
                className="transition-all duration-200 hover:brightness-110 cursor-pointer"
                style={{ filter: "url(#pie-shadow)" }}
              >
                <title>{sliceTip}</title>
              </path>
            );
          })}
        </svg>
        {/* Center label showing total */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none"
        >
          <span
            className={`font-bold leading-none text-foreground ${
              isExpanded ? "text-xl" : "text-base"
            }`}
          >
            {formatChartValue(total)}
          </span>
          <span
            className={`text-muted-foreground mt-0.5 ${
              isExpanded ? "text-xs" : "text-[10px]"
            }`}
          >
            total
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className={isExpanded ? "space-y-2.5" : "space-y-1.5"}>
        {slices.map((slice, i) => (
          <div
            key={i}
            className={`flex items-center gap-2.5 ${
              isExpanded ? "text-sm" : "text-xs"
            }`}
          >
            <div
              className={`rounded-full shrink-0 ${
                isExpanded ? "h-3 w-3" : "h-2.5 w-2.5"
              }`}
              style={{ backgroundColor: slice.color }}
            />
            <span
              className={`text-foreground/80 truncate ${
                isExpanded ? "max-w-[200px]" : "max-w-[130px]"
              }`}
              title={slice.label}
            >
              {slice.label}
            </span>
            <span
              className={`text-muted-foreground ml-auto tabular-nums ${
                isExpanded ? "text-xs" : "text-[10px]"
              }`}
            >
              {formatChartValueFull(slice.value)}
            </span>
            <span
              className={`text-muted-foreground/60 tabular-nums ${
                isExpanded ? "text-xs" : "text-[10px]"
              }`}
            >
              ({slice.percentage}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SimpleChart({
  chart_type,
  title,
  labels,
  values,
}: SimpleChartProps): React.ReactElement {
  const resolvedColors = getDefaultChartColors(chart_type);
  const [copied, setCopied] = useState(false);
  const [isSpotlightOpen, setIsSpotlightOpen] = useState(false);
  const chartContainerRef = useRef<HTMLDivElement | null>(null);

  const renderChart = useCallback(
    (isExpanded: boolean) => (
      <>
        {chart_type === "bar" && (
          <BarChart
            labels={labels}
            values={values}
            colors={resolvedColors}
            isExpanded={isExpanded}
          />
        )}
        {chart_type === "line" && (
          <LineChart
            labels={labels}
            values={values}
            colors={resolvedColors}
            isExpanded={isExpanded}
          />
        )}
        {chart_type === "pie" && (
          <PieChart
            labels={labels}
            values={values}
            colors={resolvedColors}
            isExpanded={isExpanded}
          />
        )}
      </>
    ),
    [chart_type, labels, values, resolvedColors],
  );

  const handleCopyChart = useCallback(async () => {
    const svgElement = chartContainerRef.current?.querySelector("svg");
    if (!svgElement) {
      return;
    }

    try {
      const pngBlob = await svgToPngBlob(svgElement);

      if (
        navigator.clipboard &&
        typeof ClipboardItem !== "undefined" &&
        ClipboardItem.supports?.("image/png")
      ) {
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": pngBlob }),
        ]);
      } else {
        // Fallback for environments without image clipboard support.
        const downloadUrl = URL.createObjectURL(pngBlob);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = `${title || "chart"}.png`;
        a.click();
        URL.revokeObjectURL(downloadUrl);
      }

      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can fail based on browser permissions or context.
      setCopied(false);
    }
  }, [title]);

  useEffect(() => {
    if (!isSpotlightOpen) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsSpotlightOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isSpotlightOpen]);

  return (
    <>
      <Card className="border-primary/20">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setIsSpotlightOpen(true)}
              aria-label="Open chart spotlight"
              className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={handleCopyChart}
              aria-label={copied ? "Copied chart image" : "Copy chart image"}
              className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              {copied ? (
                <Check className="h-4 w-4 text-emerald-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent ref={chartContainerRef}>{renderChart(false)}</CardContent>
      </Card>

      {isSpotlightOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label={`${title} chart spotlight`}
          onClick={() => setIsSpotlightOpen(false)}
        >
          <Card
            className="w-full max-w-5xl border-primary/30 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-base font-semibold text-foreground">
                  {title}
                </h3>
                <button
                  type="button"
                  onClick={() => setIsSpotlightOpen(false)}
                  aria-label="Close chart spotlight"
                  className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <Minimize2 className="h-4 w-4" />
                </button>
              </div>
            </CardHeader>
            <CardContent className="min-h-[380px]">{renderChart(true)}</CardContent>
          </Card>
        </div>
      )}
    </>
  );
}

export default SimpleChart;
