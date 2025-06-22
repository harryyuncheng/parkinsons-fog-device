"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

const ModelPerformance = () => {
  const [animationProgress, setAnimationProgress] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setAnimationProgress((prev) => (prev < 100 ? prev + 2 : 100));
    }, 50);
    return () => clearInterval(timer);
  }, []);

  // Model performance data based on your results
  const modelStats = {
    totalSamples: 68247,
    sequences: 1065,
    sequenceLength: 128,
    totalParams: 349379,
    testAccuracy: 96.27,
    bestValAccuracy: 99.37,
    trainingEpochs: 100,
  };

  const classPerformance = [
    {
      name: "Walking",
      precision: 0.98,
      recall: 0.97,
      f1: 0.98,
      support: 63,
      color: "bg-blue-500",
    },
    {
      name: "Freezing",
      precision: 0.97,
      recall: 0.97,
      f1: 0.97,
      support: 71,
      color: "bg-red-500",
    },
    {
      name: "Standing",
      precision: 0.89,
      recall: 0.93,
      f1: 0.91,
      support: 27,
      color: "bg-green-500",
    },
  ];

  const dataDistribution = [
    { label: "Walking", count: 26997, percentage: 39.5, color: "bg-blue-500" },
    { label: "Freezing", count: 22233, percentage: 32.6, color: "bg-red-500" },
    {
      label: "Standing",
      count: 19017,
      percentage: 27.9,
      color: "bg-green-500",
    },
  ];

  const architectureLayers = [
    { name: "Input", type: "IMU Data", size: "6 channels" },
    { name: "Conv1D-1", type: "CNN", size: "64 filters" },
    { name: "Conv1D-2", type: "CNN", size: "128 filters" },
    { name: "Conv1D-3", type: "CNN", size: "256 filters" },
    { name: "LSTM", type: "RNN", size: "128 hidden" },
    { name: "Dense", type: "FC", size: "64 neurons" },
    { name: "Output", type: "Classification", size: "3 classes" },
  ];

  const NeuralNetworkViz = () => (
    <div className="relative h-48 w-full overflow-hidden bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-lg">
      {/* Neural network nodes */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 200">
        {/* Input layer */}
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <circle
            key={`input-${i}`}
            cx="40"
            cy={30 + i * 25}
            r="6"
            fill="#3b82f6"
            opacity={animationProgress > i * 10 ? 1 : 0.3}
            className="transition-opacity duration-500"
          />
        ))}

        {/* Hidden layers */}
        {[0, 1, 2, 3].map((i) => (
          <circle
            key={`hidden1-${i}`}
            cx="120"
            cy={50 + i * 30}
            r="8"
            fill="#8b5cf6"
            opacity={animationProgress > 30 + i * 5 ? 1 : 0.3}
            className="transition-opacity duration-500"
          />
        ))}

        {[0, 1, 2, 3, 4].map((i) => (
          <circle
            key={`hidden2-${i}`}
            cx="200"
            cy={40 + i * 30}
            r="8"
            fill="#10b981"
            opacity={animationProgress > 50 + i * 5 ? 1 : 0.3}
            className="transition-opacity duration-500"
          />
        ))}

        {/* LSTM layer */}
        <rect
          x="280"
          y="80"
          width="30"
          height="40"
          rx="4"
          fill="#f59e0b"
          opacity={animationProgress > 70 ? 1 : 0.3}
          className="transition-opacity duration-500"
        />

        {/* Output layer */}
        {[0, 1, 2].map((i) => (
          <circle
            key={`output-${i}`}
            cx="360"
            cy={70 + i * 30}
            r="10"
            fill="#ef4444"
            opacity={animationProgress > 80 + i * 5 ? 1 : 0.3}
            className="transition-opacity duration-500"
          />
        ))}

        {/* Connections (simplified) */}
        {animationProgress > 90 && (
          <>
            <line
              x1="48"
              y1="80"
              x2="112"
              y2="80"
              stroke="#4f46e5"
              strokeWidth="1"
              opacity="0.6"
            />
            <line
              x1="128"
              y1="80"
              x2="192"
              y2="80"
              stroke="#7c3aed"
              strokeWidth="1"
              opacity="0.6"
            />
            <line
              x1="208"
              y1="100"
              x2="280"
              y2="100"
              stroke="#059669"
              strokeWidth="1"
              opacity="0.6"
            />
            <line
              x1="310"
              y1="100"
              x2="350"
              y2="100"
              stroke="#dc2626"
              strokeWidth="1"
              opacity="0.6"
            />
          </>
        )}
      </svg>

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full animate-pulse"
            style={{
              left: `${20 + i * 60}%`,
              top: `${30 + (i % 3) * 30}%`,
              animationDelay: `${i * 200}ms`,
            }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          🧠 FOG Detection AI Model
        </h1>
        <p className="text-gray-600 text-lg">
          CNN-LSTM Hybrid Architecture Performance Dashboard
        </p>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
          <CardContent className="p-6 text-center">
            <div className="text-3xl font-bold text-blue-600">
              {modelStats.testAccuracy}%
            </div>
            <div className="text-sm text-blue-700 font-medium">
              Test Accuracy
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
          <CardContent className="p-6 text-center">
            <div className="text-3xl font-bold text-green-600">
              {modelStats.bestValAccuracy}%
            </div>
            <div className="text-sm text-green-700 font-medium">
              Best Val Accuracy
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
          <CardContent className="p-6 text-center">
            <div className="text-3xl font-bold text-purple-600">
              {modelStats.totalParams.toLocaleString()}
            </div>
            <div className="text-sm text-purple-700 font-medium">
              Parameters
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200">
          <CardContent className="p-6 text-center">
            <div className="text-3xl font-bold text-orange-600">
              {modelStats.totalSamples.toLocaleString()}
            </div>
            <div className="text-sm text-orange-700 font-medium">
              Training Samples
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Neural Network Visualization */}
        <Card className="col-span-1 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🔬 Model Architecture
              <Badge variant="secondary">CNN-LSTM Hybrid</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <NeuralNetworkViz />
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
              {architectureLayers.map((layer, idx) => (
                <div key={idx} className="text-center p-2 bg-gray-50 rounded">
                  <div className="font-semibold">{layer.name}</div>
                  <div className="text-gray-600">{layer.type}</div>
                  <div className="text-gray-500">{layer.size}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Class Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🎯 Classification Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {classPerformance.map((cls, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${cls.color}`} />
                    {cls.name}
                  </span>
                  <Badge variant="outline">Support: {cls.support}</Badge>
                </div>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <div className="text-gray-600">Precision</div>
                    <Progress value={cls.precision * 100} className="h-2" />
                    <div className="text-xs text-right mt-1">
                      {(cls.precision * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">Recall</div>
                    <Progress value={cls.recall * 100} className="h-2" />
                    <div className="text-xs text-right mt-1">
                      {(cls.recall * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-600">F1-Score</div>
                    <Progress value={cls.f1 * 100} className="h-2" />
                    <div className="text-xs text-right mt-1">
                      {(cls.f1 * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Data Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              📊 Training Data Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {dataDistribution.map((data, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${data.color}`} />
                    {data.label}
                  </span>
                  <span className="text-sm text-gray-600">
                    {data.count.toLocaleString()}
                  </span>
                </div>
                <div className="relative">
                  <Progress value={data.percentage} className="h-3" />
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                    {data.percentage.toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}

            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Total Sequences:</span>
                  <span className="font-medium">{modelStats.sequences}</span>
                </div>
                <div className="flex justify-between">
                  <span>Sequence Length:</span>
                  <span className="font-medium">
                    {modelStats.sequenceLength} timesteps
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Training Epochs:</span>
                  <span className="font-medium">
                    {modelStats.trainingEpochs}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Training Progress Simulation
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            📈 Training Progress Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-medium text-gray-700">Model Convergence</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Initial Accuracy</span>
                  <span>85.77%</span>
                </div>
                <Progress value={85.77} className="h-2" />

                <div className="flex justify-between text-sm">
                  <span>Final Validation Accuracy</span>
                  <span>99.37%</span>
                </div>
                <Progress value={99.37} className="h-2" />

                <div className="flex justify-between text-sm">
                  <span>Test Accuracy</span>
                  <span>96.27%</span>
                </div>
                <Progress value={96.27} className="h-2" />
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium text-gray-700">Key Achievements</h4>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Excellent freezing detection (97% F1-score)</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>High walking accuracy (98% precision)</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span>Robust CNN-LSTM architecture</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  <span>Triple prediction buffer for stability</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card> */}
    </div>
  );
};

export default ModelPerformance;
