import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Network,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Info,
  Filter,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import { api } from "../lib/api";
import { RandomForestTreeResponse, TreeNode } from "../types/api";

interface NodePosition {
  x: number;
  y: number;
}

export const TreeExplorerSection: React.FC = () => {
  const [treeIndex, setTreeIndex] = useState<number>(0);
  const [maxDepth, setMaxDepth] = useState<number | null>(3); // Default to readable depth 3, with Full Tree option
  const [treeData, setTreeData] = useState<RandomForestTreeResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Canvas zoom & pan state
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 50, y: 50 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch tree data whenever treeIndex or maxDepth changes
  useEffect(() => {
    let isMounted = true;
    const fetchTree = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.getTree(treeIndex, maxDepth === null ? undefined : maxDepth);
        if (isMounted) {
          setTreeData(res);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message || "Failed to load decision tree structure.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchTree();
    return () => {
      isMounted = false;
    };
  }, [treeIndex, maxDepth]);

  // Compute Layout coordinates for nodes
  const NODE_WIDTH = 220;
  const NODE_HEIGHT = 135;
  const LEVEL_HEIGHT = 190;
  const SIBLING_GAP = 30;

  const layout = useMemo(() => {
    if (!treeData || !treeData.nodes.length) {
      return { positions: new Map<number, NodePosition>(), width: 800, height: 600 };
    }

    const nodeMap = new Map<number, TreeNode>();
    treeData.nodes.forEach((n) => nodeMap.set(n.id, n));

    const positions = new Map<number, NodePosition>();
    let leafCounter = 0;

    // Subtree layout: leaves get sequential x slots, parents are centered above children
    const calculatePositions = (nodeId: number, depth: number): number => {
      const node = nodeMap.get(nodeId);
      if (!node) return 0;

      const hasLeft = node.left_child != null && nodeMap.has(node.left_child);
      const hasRight = node.right_child != null && nodeMap.has(node.right_child);

      if (node.is_leaf || (!hasLeft && !hasRight)) {
        const x = leafCounter * (NODE_WIDTH + SIBLING_GAP);
        leafCounter++;
        positions.set(nodeId, { x, y: depth * LEVEL_HEIGHT });
        return x;
      }

      const leftX = hasLeft ? calculatePositions(node.left_child!, depth + 1) : null;
      const rightX = hasRight ? calculatePositions(node.right_child!, depth + 1) : null;

      let x: number;
      if (leftX != null && rightX != null) {
        x = (leftX + rightX) / 2;
      } else if (leftX != null) {
        x = leftX;
      } else if (rightX != null) {
        x = rightX;
      } else {
        x = leafCounter * (NODE_WIDTH + SIBLING_GAP);
        leafCounter++;
      }

      positions.set(nodeId, { x, y: depth * LEVEL_HEIGHT });
      return x;
    };

    calculatePositions(0, 0);

    let maxX = 800;
    let maxY = 600;
    positions.forEach((pos) => {
      if (pos.x + NODE_WIDTH + 100 > maxX) maxX = pos.x + NODE_WIDTH + 100;
      if (pos.y + NODE_HEIGHT + 100 > maxY) maxY = pos.y + NODE_HEIGHT + 100;
    });

    return { positions, width: maxX, height: maxY };
  }, [treeData]);

  // Reset zoom & fit to screen
  const handleFitScreen = () => {
    if (!containerRef.current || layout.width === 0) return;
    const containerWidth = containerRef.current.clientWidth;
    const fitScale = Math.min(Math.max(containerWidth / (layout.width + 100), 0.35), 1.0);
    setZoom(fitScale);
    setPan({ x: 30, y: 30 });
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.15, 1.8));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.15, 0.25));
  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 40, y: 40 });
  };

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Left mouse button only
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div id="tree-explorer" className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
      {/* 1. Header & Summary Stats */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h3 className="flex items-center gap-2 text-base font-semibold text-slate-900">
            <Network className="h-5 w-5 text-brand-600" />
            Random Forest Tree Explorer
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            Interactive structural inspection of real decision trees inside the trained 150-estimator ensemble.
          </p>
        </div>

        {/* Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
            Total Trees: <strong className="ml-1 text-slate-900">{treeData?.total_estimators || 150}</strong>
          </span>
          <span className="inline-flex items-center rounded-md bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800">
            Selected: <strong className="ml-1">Tree {treeIndex + 1}</strong>
          </span>
          <span className="inline-flex items-center rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
            Nodes: <strong className="ml-1 text-slate-900">{treeData?.node_count ?? "..."}</strong>
          </span>
          <span className="inline-flex items-center rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
            Max Depth: <strong className="ml-1 text-slate-900">{treeData?.max_depth ?? "..."}</strong>
          </span>
        </div>
      </div>

      {/* 2. Interactive Navigation & View Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-50/70 p-3 rounded-lg border border-slate-200 text-xs">
        {/* Left: Tree Selector */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setTreeIndex((idx) => Math.max(0, idx - 1))}
            disabled={treeIndex <= 0 || loading}
            className="flex items-center gap-1 rounded border border-slate-300 bg-white px-2.5 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40 shadow-xs"
            title="Previous Tree"
          >
            <ChevronLeft className="h-4 w-4" />
            Prev Tree
          </button>

          <div className="flex items-center gap-1.5 font-medium text-slate-700">
            <span>Tree</span>
            <select
              value={treeIndex}
              onChange={(e) => setTreeIndex(Number(e.target.value))}
              disabled={loading}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-semibold text-slate-900 shadow-xs focus:ring-1 focus:ring-brand-500 focus:outline-hidden"
            >
              {Array.from({ length: treeData?.total_estimators || 150 }, (_, i) => (
                <option key={i} value={i}>
                  Tree {i + 1} of {treeData?.total_estimators || 150}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => setTreeIndex((idx) => Math.min((treeData?.total_estimators || 150) - 1, idx + 1))}
            disabled={treeIndex >= (treeData?.total_estimators || 150) - 1 || loading}
            className="flex items-center gap-1 rounded border border-slate-300 bg-white px-2.5 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40 shadow-xs"
            title="Next Tree"
          >
            Next Tree
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Center: Depth Filter (Exploration Aid) */}
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 font-medium text-slate-600">
            <Filter className="h-3.5 w-3.5 text-slate-400" />
            View Depth:
          </span>
          <div className="flex items-center rounded border border-slate-300 bg-white p-0.5 shadow-xs">
            <button
              onClick={() => setMaxDepth(2)}
              className={`px-2 py-1 rounded text-xs ${
                maxDepth === 2 ? "bg-slate-900 text-white font-semibold" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              2
            </button>
            <button
              onClick={() => setMaxDepth(3)}
              className={`px-2 py-1 rounded text-xs ${
                maxDepth === 3 ? "bg-slate-900 text-white font-semibold" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              3 (Overview)
            </button>
            <button
              onClick={() => setMaxDepth(4)}
              className={`px-2 py-1 rounded text-xs ${
                maxDepth === 4 ? "bg-slate-900 text-white font-semibold" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              4
            </button>
            <button
              onClick={() => setMaxDepth(5)}
              className={`px-2 py-1 rounded text-xs ${
                maxDepth === 5 ? "bg-slate-900 text-white font-semibold" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              5
            </button>
            <button
              onClick={() => setMaxDepth(null)}
              className={`px-2 py-1 rounded text-xs font-semibold ${
                maxDepth === null ? "bg-brand-600 text-white" : "text-slate-700 hover:bg-slate-100"
              }`}
            >
              Full Tree ({treeData?.max_depth || 12} Levels)
            </button>
          </div>
        </div>

        {/* Right: Zoom & Pan Controls */}
        <div className="flex items-center gap-1">
          <span className="mr-1 text-slate-500 font-mono text-[11px]">{Math.round(zoom * 100)}%</span>
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 shadow-xs"
            title="Zoom In"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 shadow-xs"
            title="Zoom Out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            onClick={handleFitScreen}
            className="p-1.5 rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 shadow-xs"
            title="Fit to Screen"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
          <button
            onClick={handleResetZoom}
            className="p-1.5 rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 shadow-xs"
            title="Reset Pan & Zoom"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* 3. Main Tree Canvas */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="relative h-[550px] w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-900/5 select-none cursor-grab active:cursor-grabbing"
        style={{
          backgroundImage: "radial-gradient(#cbd5e1 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      >
        {loading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/70 backdrop-blur-xs">
            <div className="h-8 w-8 animate-spin rounded-full border-3 border-brand-600 border-t-transparent mb-2" />
            <span className="text-xs font-medium text-slate-600">Extracting tree structure from Random Forest...</span>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 z-20 flex items-center justify-center p-6 bg-white/90">
            <div className="max-w-md rounded-lg border border-rose-200 bg-rose-50 p-4 text-center">
              <AlertTriangle className="h-6 w-6 text-rose-600 mx-auto mb-2" />
              <p className="text-xs font-semibold text-rose-900">Failed to load tree</p>
              <p className="text-xs text-rose-700 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Transformable Canvas Layer */}
        <div
          className="absolute origin-top-left transition-transform duration-75 ease-out"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            width: `${layout.width}px`,
            height: `${layout.height}px`,
          }}
        >
          {/* SVG Connector Lines Layer */}
          <svg
            className="absolute top-0 left-0 pointer-events-none"
            style={{ width: layout.width, height: layout.height }}
          >
            {treeData?.edges.map((edge, idx) => {
              const srcPos = layout.positions.get(edge.source);
              const tgtPos = layout.positions.get(edge.target);
              if (!srcPos || !tgtPos) return null;

              const x1 = srcPos.x + NODE_WIDTH / 2;
              const y1 = srcPos.y + NODE_HEIGHT;
              const x2 = tgtPos.x + NODE_WIDTH / 2;
              const y2 = tgtPos.y;

              const midY = (y1 + y2) / 2;
              const pathD = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
              const isLeft = edge.branch === "left";

              return (
                <g key={`edge-${idx}`}>
                  <path
                    d={pathD}
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="1.75"
                    strokeDasharray={isLeft ? undefined : undefined}
                  />
                  {/* Midpoint condition badge */}
                  <rect
                    x={(x1 + x2) / 2 - 45}
                    y={midY - 10}
                    width={90}
                    height={20}
                    rx={4}
                    fill="#ffffff"
                    stroke="#cbd5e1"
                    strokeWidth="1"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={midY + 3.5}
                    textAnchor="middle"
                    className="text-[10px] font-semibold fill-slate-700 font-mono"
                  >
                    {edge.condition}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Node Cards Layer */}
          {treeData?.nodes.map((node) => {
            const pos = layout.positions.get(node.id);
            if (!pos) return null;

            const isFailurePred = node.predicted_class === 1;
            const normCount = node.class_counts[0] ?? 0;
            const failCount = node.class_counts[1] ?? 0;
            const totalSamples = normCount + failCount || node.samples;
            const failPct = totalSamples > 0 ? (failCount / totalSamples) * 100 : 0;

            return (
              <div
                key={`node-${node.id}`}
                className={`absolute rounded-xl border p-3 shadow-sm transition-shadow hover:shadow-md bg-white ${
                  node.is_leaf
                    ? isFailurePred
                      ? "border-rose-300 ring-2 ring-rose-100 bg-rose-50/20"
                      : "border-emerald-300 ring-2 ring-emerald-100 bg-emerald-50/20"
                    : "border-slate-300 hover:border-slate-400"
                }`}
                style={{
                  left: `${pos.x}px`,
                  top: `${pos.y}px`,
                  width: `${NODE_WIDTH}px`,
                  height: `${NODE_HEIGHT}px`,
                }}
              >
                {/* Node Top Bar */}
                <div className="flex items-center justify-between border-b border-slate-100 pb-1.5 mb-1.5 text-[10px]">
                  <span className="font-mono text-slate-400">
                    {node.id === 0 ? "ROOT" : node.is_leaf ? `LEAF #${node.id}` : `NODE #${node.id}`}
                  </span>
                  <span className="text-slate-500 font-medium">Gini: {node.gini.toFixed(3)}</span>
                  <span className="font-medium text-slate-700">{node.samples.toLocaleString()} smp</span>
                </div>

                {/* Node Body Content */}
                {node.is_leaf ? (
                  /* Leaf Node Presentation */
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-medium text-slate-600">Decision:</span>
                      <span
                        className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-bold ${
                          isFailurePred
                            ? "bg-rose-100 text-rose-800"
                            : "bg-emerald-100 text-emerald-800"
                        }`}
                      >
                        {isFailurePred ? (
                          <>
                            <AlertTriangle className="h-3 w-3" />
                            FAILURE
                          </>
                        ) : (
                          <>
                            <CheckCircle className="h-3 w-3" />
                            NORMAL
                          </>
                        )}
                      </span>
                    </div>

                    {/* Class Distribution Mini Bar */}
                    <div className="pt-1">
                      <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                        <span>Normal: {normCount}</span>
                        <span className={isFailurePred ? "font-bold text-rose-700" : ""}>
                          Fail: {failCount} ({failPct.toFixed(1)}%)
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden flex">
                        <div
                          className="bg-emerald-500 h-full"
                          style={{ width: `${100 - failPct}%` }}
                        />
                        <div
                          className="bg-rose-500 h-full"
                          style={{ width: `${failPct}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Decision Node Presentation */
                  <div className="space-y-1.5">
                    <div>
                      <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-tight truncate">
                        {node.feature_label || node.feature}
                      </div>
                      <div className="text-xs font-bold text-slate-900 font-mono mt-0.5">
                        {node.condition_left || `≤ ${node.threshold}`}
                      </div>
                    </div>

                    {/* Distribution preview */}
                    <div className="pt-1">
                      <div className="flex justify-between text-[10px] text-slate-500 mb-0.5">
                        <span>N: {normCount}</span>
                        <span>F: {failCount}</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden flex">
                        <div
                          className="bg-emerald-500 h-full"
                          style={{ width: `${100 - failPct}%` }}
                        />
                        <div
                          className="bg-rose-500 h-full"
                          style={{ width: `${failPct}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. Legend & Academic "How to Read the Tree" Box */}
      <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-2">
        <h4 className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
          <Info className="h-4 w-4 text-brand-600" />
          How to Read the Decision Tree
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-slate-600 leading-relaxed">
          <div>
            <p>
              • <strong>Decision Nodes:</strong> Each decision node applies a learned split on a feature (e.g., Spindle Speed, Mechanical Power, Wear Load).
            </p>
            <p className="mt-1">
              • <strong>Branching Logic:</strong> The left branch corresponds to the threshold comparison (<code>≤ threshold</code>); the right branch corresponds to <code>&gt; threshold</code>.
            </p>
          </div>
          <div>
            <p>
              • <strong>Leaf Nodes:</strong> Each leaf contains the model's final class distribution and predicted status (Normal or Failure) for that partitioned region.
            </p>
            <p className="mt-1">
              • <strong>Ensemble Architecture:</strong> A Random Forest consists of many such trees (150 in MachineGuard AI). The final prediction combines all trees through soft voting rather than relying on any single tree. This explorer displays one actual constituent tree from the trained model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
