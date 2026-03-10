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
  colors?: string[];
}

const DEFAULT_COLORS = [
  "#003B6F",
  "#FFD100",
  "#3EB1C8",
  "#64748b",
  "#f43f5e",
  "#8b5cf6",
];

/**
 * Formats chart numeric values for readability.
 */
function formatChartValue(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
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
  const svgWidth = labels.length * (barWidth + gap) + gap;

  const truncateLabel = (label: string) => {
    if (label.length <= 6) return label;
    return label.slice(0, 5) + "...";
  };

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${svgWidth} ${chartHeight + 40}`}
      className="overflow-visible"
    >
      {values.map((val, i) => {
        const barHeight = (val / maxVal) * chartHeight;
        const x = gap + i * (barWidth + gap);
        const y = chartHeight - barHeight;
        const color = colors[i % colors.length];

        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx={4}
              fill={color}
              opacity={0.85}
              className="transition-all duration-300 hover:opacity-100"
            />
            <text
              x={x + barWidth / 2}
              y={y - (isExpanded ? 8 : 6)}
              textAnchor="middle"
              fontSize={isExpanded ? "11" : "9"}
              fontWeight="500"
              className="fill-current opacity-70"
              style={{ fill: "currentColor" }}
            >
              {formatChartValue(val)}
            </text>
            <text
              x={x + barWidth / 2}
              y={chartHeight + (isExpanded ? 20 : 16)}
              textAnchor="middle"
              fontSize={isExpanded ? "9" : "8"}
              className="fill-current opacity-60 cursor-help"
              style={{ fill: "currentColor" }}
            >
              <title>{labels[i]}</title>
              {truncateLabel(labels[i])}
            </text>
          </g>
        );
      })}
      <line
        x1={0}
        y1={chartHeight}
        x2={svgWidth}
        y2={chartHeight}
        stroke="currentColor"
        strokeOpacity={0.15}
      />
    </svg>
  );
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
  const padding = isExpanded ? 28 : 20;
  const svgWidth = Math.max(labels.length * (isExpanded ? 72 : 60), 300);
  const stepX = (svgWidth - 2 * padding) / Math.max(labels.length - 1, 1);
  const color = colors[0] || DEFAULT_COLORS[0];

  const points = values.map((val, i) => ({
    x: padding + i * stepX,
    y: chartHeight - (val / maxVal) * (chartHeight - 20),
  }));

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  const truncateLabel = (label: string) => {
    if (label.length <= 8) return label;
    return label.slice(0, 7) + "...";
  };

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${svgWidth} ${chartHeight + 40}`}
      className="overflow-visible"
    >
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth={isExpanded ? 3 : 2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {points.map((p, i) => (
        <g key={i}>
          <circle
            cx={p.x}
            cy={p.y}
            r={isExpanded ? 5 : 4}
            fill={color}
            className="transition-all duration-200"
          />
          <text
            x={p.x}
            y={p.y - (isExpanded ? 12 : 10)}
            textAnchor="middle"
            fontSize={isExpanded ? "11" : "9"}
            fontWeight="500"
            className="fill-current opacity-70"
            style={{ fill: "currentColor" }}
          >
            {formatChartValue(values[i])}
          </text>
          <text
            x={p.x}
            y={chartHeight + (isExpanded ? 20 : 16)}
            textAnchor="middle"
            fontSize={isExpanded ? "9" : "8"}
            className="fill-current opacity-60 cursor-help"
            style={{ fill: "currentColor" }}
          >
            <title>{labels[i]}</title>
            {truncateLabel(labels[i])}
          </text>
        </g>
      ))}

      <line
        x1={0}
        y1={chartHeight}
        x2={svgWidth}
        y2={chartHeight}
        stroke="currentColor"
        strokeOpacity={0.15}
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
  const size = isExpanded ? 240 : 180;
  const cx = size / 2;
  const cy = size / 2;
  const radius = isExpanded ? 92 : 70;

  let cumulative = 0;
  const slices = values.map((val, i) => {
    const startAngle = (cumulative / total) * 2 * Math.PI - Math.PI / 2;
    cumulative += val;
    const endAngle = (cumulative / total) * 2 * Math.PI - Math.PI / 2;

    const x1 = cx + radius * Math.cos(startAngle);
    const y1 = cy + radius * Math.sin(startAngle);
    const x2 = cx + radius * Math.cos(endAngle);
    const y2 = cy + radius * Math.sin(endAngle);
    const largeArc = val / total > 0.5 ? 1 : 0;

    return {
      path: `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z`,
      color: colors[i % colors.length],
      label: labels[i],
      percentage: ((val / total) * 100).toFixed(0),
    };
  });

  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-center md:gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {slices.map((slice, i) => (
          <path
            key={i}
            d={slice.path}
            fill={slice.color}
            opacity={0.85}
            className="transition-opacity hover:opacity-100"
          />
        ))}
      </svg>
      <div className={isExpanded ? "space-y-2" : "space-y-1.5"}>
        {slices.map((slice, i) => (
          <div
            key={i}
            className={`flex items-center gap-2 ${
              isExpanded ? "text-sm" : "text-xs"
            }`}
          >
            <div
              className={`rounded-sm ${
                isExpanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"
              }`}
              style={{ backgroundColor: slice.color }}
            />
            <span
              className={`text-current opacity-80 truncate ${
                isExpanded ? "max-w-[180px]" : "max-w-[120px]"
              }`}
              title={slice.label}
            >
              {slice.label}
            </span>
            <span className={isExpanded ? "text-current opacity-60 text-xs" : "text-current opacity-60 text-[10px]"}>
              {slice.percentage}%
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
  colors,
}: SimpleChartProps): React.ReactElement {
  const resolvedColors =
    colors && colors.length > 0 ? colors : DEFAULT_COLORS;
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
