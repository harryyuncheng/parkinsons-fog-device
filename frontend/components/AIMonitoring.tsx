import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Brain, RotateCcw, AlertTriangle } from "lucide-react";
import { api } from "@/lib/api";

interface PredictionData {
  prediction: string;
  raw_prediction?: string;
  confidence: number;
  probabilities: {
    walking: number;
    standing: number;
    freezing: number;
  };
  buffer_size: number;
  status: string;
}

interface AIMonitoringProps {
  isConnected: boolean;
  realTimePrediction?: PredictionData;
}

// Single Animated Stick Figure Component that changes based on state
const AnimatedStickFigure = ({ currentState }: { currentState: string }) => {
  const [walkStep, setWalkStep] = useState(0);
  const [shake, setShake] = useState(0);

  useEffect(() => {
    if (currentState === "walking") {
      const interval = setInterval(() => {
        setWalkStep((prev) => (prev + 1) % 4);
      }, 400);
      return () => clearInterval(interval);
    }
  }, [currentState]);

  useEffect(() => {
    if (currentState === "freezing") {
      const interval = setInterval(() => {
        setShake((prev) => (prev + 1) % 8);
      }, 100);
      return () => clearInterval(interval);
    }
  }, [currentState]);

  // Animation configurations for walking
  const legPositions = [
    { left: { x2: 30, y2: 105 }, right: { x2: 50, y2: 105 } }, // neutral
    { left: { x2: 35, y2: 105 }, right: { x2: 45, y2: 105 } }, // left forward
    { left: { x2: 30, y2: 105 }, right: { x2: 50, y2: 105 } }, // neutral
    { left: { x2: 25, y2: 105 }, right: { x2: 55, y2: 105 } }, // right forward
  ];

  const armPositions = [
    { left: { x2: 25, y2: 50 }, right: { x2: 55, y2: 50 } },
    { left: { x2: 20, y2: 45 }, right: { x2: 60, y2: 55 } },
    { left: { x2: 25, y2: 50 }, right: { x2: 55, y2: 50 } },
    { left: { x2: 30, y2: 55 }, right: { x2: 50, y2: 45 } },
  ];

  // Color and animation settings based on state
  const getStateConfig = () => {
    switch (currentState) {
      case "walking":
        return {
          color: "#22c55e",
          textColor: "text-green-600",
          label: "Walking",
        };
      case "freezing":
        return {
          color: "#ef4444",
          textColor: "text-red-600",
          label: "Freezing",
        };
      case "standing":
      default:
        return {
          color: "#3b82f6",
          textColor: "text-blue-600",
          label: "Standing",
        };
    }
  };

  const config = getStateConfig();
  const shakeOffset =
    currentState === "freezing" ? (shake % 2 === 0 ? 1 : -1) : 0;

  return (
    <div className="transition-all duration-300 scale-110">
      <svg width="120" height="160" viewBox="0 0 120 160" className="mx-auto">
        {/* Head */}
        <circle
          cx={60 + (currentState === "freezing" ? shakeOffset : 0)}
          cy={20 + (currentState === "freezing" ? shakeOffset * 0.5 : 0)}
          r="12"
          fill={config.color}
        />

        {/* Body */}
        <line
          x1={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y1={32 + (currentState === "freezing" ? shakeOffset * 0.5 : 0)}
          x2={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y2={90 + (currentState === "freezing" ? shakeOffset * 0.3 : 0)}
          stroke={config.color}
          strokeWidth="4"
        />

        {/* Arms */}
        <line
          x1={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y1={45 + (currentState === "freezing" ? shakeOffset * 0.3 : 0)}
          x2={
            currentState === "walking"
              ? 40 + armPositions[walkStep].left.x2 - 25
              : 40 + (currentState === "freezing" ? shakeOffset * 2 : 0)
          }
          y2={
            currentState === "walking"
              ? armPositions[walkStep].left.y2 + 15
              : 65 + (currentState === "freezing" ? shakeOffset : 0)
          }
          stroke={config.color}
          strokeWidth="3"
          className="transition-all duration-200"
        />
        <line
          x1={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y1={45 + (currentState === "freezing" ? shakeOffset * 0.3 : 0)}
          x2={
            currentState === "walking"
              ? 40 + armPositions[walkStep].right.x2 - 25
              : 80 + (currentState === "freezing" ? shakeOffset * 2 : 0)
          }
          y2={
            currentState === "walking"
              ? armPositions[walkStep].right.y2 + 15
              : 65 + (currentState === "freezing" ? shakeOffset : 0)
          }
          stroke={config.color}
          strokeWidth="3"
          className="transition-all duration-200"
        />

        {/* Legs */}
        <line
          x1={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y1={90 + (currentState === "freezing" ? shakeOffset * 0.3 : 0)}
          x2={
            currentState === "walking"
              ? 30 + legPositions[walkStep].left.x2 - 30
              : 45 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y2={
            currentState === "walking"
              ? legPositions[walkStep].left.y2 + 25
              : 130
          }
          stroke={config.color}
          strokeWidth="4"
          className="transition-all duration-200"
        />
        <line
          x1={60 + (currentState === "freezing" ? shakeOffset : 0)}
          y1={90 + (currentState === "freezing" ? shakeOffset * 0.3 : 0)}
          x2={
            currentState === "walking"
              ? 30 + legPositions[walkStep].right.x2 - 30
              : 75 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y2={
            currentState === "walking"
              ? legPositions[walkStep].right.y2 + 25
              : 130
          }
          stroke={config.color}
          strokeWidth="4"
          className="transition-all duration-200"
        />

        {/* Feet */}
        <line
          x1={
            currentState === "walking"
              ? legPositions[walkStep].left.x2
              : 45 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y1="130"
          x2={
            currentState === "walking"
              ? legPositions[walkStep].left.x2 - 8
              : 37 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y2="130"
          stroke={config.color}
          strokeWidth="3"
        />
        <line
          x1={
            currentState === "walking"
              ? legPositions[walkStep].right.x2 + 25
              : 75 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y1="130"
          x2={
            currentState === "walking"
              ? legPositions[walkStep].right.x2 + 33
              : 83 + (currentState === "freezing" ? shakeOffset : 0)
          }
          y2="130"
          stroke={config.color}
          strokeWidth="3"
        />

        {/* Tremor particles for freezing state */}
        {currentState === "freezing" && (
          <>
            <circle
              cx={60 + shakeOffset * 3}
              cy={75}
              r="1.5"
              fill={config.color}
              opacity="0.6"
            />
            <circle
              cx={60 - shakeOffset * 2}
              cy={55}
              r="1.5"
              fill={config.color}
              opacity="0.6"
            />
            <circle
              cx={60 + shakeOffset * 2}
              cy={95}
              r="1.5"
              fill={config.color}
              opacity="0.6"
            />
          </>
        )}
      </svg>
      <div className={`text-center text-lg font-bold mt-3 ${config.textColor}`}>
        {config.label}
      </div>
    </div>
  );
};

export default function AIMonitoring({
  isConnected,
  realTimePrediction,
}: AIMonitoringProps) {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const prevPredictionRef = useRef<PredictionData | null>(null);

  // Use real-time prediction if provided, otherwise poll manually
  useEffect(() => {
    if (realTimePrediction) {
      const prev = prevPredictionRef.current;
      const hasChanged =
        !prev ||
        prev.prediction !== realTimePrediction.prediction ||
        prev.confidence !== realTimePrediction.confidence ||
        prev.status !== realTimePrediction.status;

      if (hasChanged) {
        setPrediction(realTimePrediction);
        setLastUpdate(new Date());
        prevPredictionRef.current = realTimePrediction;
      }
    }
  }, [realTimePrediction]);

  // Fallback polling if real-time data is not available
  useEffect(() => {
    if (!realTimePrediction && isConnected) {
      const interval = setInterval(async () => {
        try {
          const result = await api.getPrediction();
          if (result.status === "success") {
            setPrediction(result.data.prediction);
            setLastUpdate(new Date());
          }
        } catch (error) {
          console.error("Error fetching prediction:", error);
        }
      }, 1000); // Poll every second

      return () => clearInterval(interval);
    }
  }, [isConnected, realTimePrediction]);

  const handleResetBuffer = async () => {
    setIsLoading(true);
    try {
      const result = await api.resetPredictor();
      if (result.status === "success") {
        console.log("Buffer reset successfully");
        setPrediction(null);
      }
    } catch (error) {
      console.error("Error resetting buffer:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return "text-green-600";
    if (confidence > 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>AI FOG Detection Monitor</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Real-time monitoring using trained CNN-LSTM model
            </div>
            <Button
              onClick={handleResetBuffer}
              disabled={!isConnected || isLoading}
              variant="outline"
              size="sm"
              className="flex items-center space-x-2"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset Buffer</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Connection Status */}
      {!isConnected && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="px-6 py-4">
            <div className="flex items-center space-x-2 text-yellow-800">
              <AlertTriangle className="w-5 h-5" />
              <span>
                Backend not connected. Please ensure ESP32 and Flask server are
                running.
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Prediction with Stick Figures */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg text-center">
            Current AI Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          {prediction ? (
            <div className="space-y-6">
              {/* Stick Figure Display */}
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-6">
                <div className="flex justify-center">
                  <AnimatedStickFigure currentState={prediction.prediction} />
                </div>

                {/* Confidence display */}
                <div className="text-center mt-4">
                  <div
                    className={`text-lg font-bold ${getConfidenceColor(
                      prediction.confidence
                    )}`}
                  >
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    {prediction.prediction === "freezing" &&
                      prediction.confidence > 0.8 &&
                      "🚨 High confidence freeze detected!"}
                    {prediction.prediction === "walking" &&
                      prediction.confidence > 0.8 &&
                      "✅ Normal walking detected"}
                    {prediction.prediction === "standing" &&
                      prediction.confidence > 0.8 &&
                      "🧍 Standing position detected"}
                  </div>
                </div>
              </div>

              {/* Probability breakdown */}
              <div className="space-y-3">
                <h4 className="font-medium">Probability Breakdown:</h4>
                {Object.entries(prediction.probabilities).map(
                  ([state, prob]) => (
                    <div key={state} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize">{state}</span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={prob * 100} className="h-2" />
                    </div>
                  )
                )}
              </div>

              {/* Status info */}
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <div className="font-medium">Status:</div>
                  <div className="capitalize">{prediction.status}</div>
                </div>
                <div>
                  <div className="font-medium">Last Update:</div>
                  <div>
                    {lastUpdate ? lastUpdate.toLocaleTimeString() : "Never"}
                  </div>
                </div>
              </div>

              {/* Raw prediction if different */}
              {prediction.raw_prediction &&
                prediction.raw_prediction !== prediction.prediction && (
                  <div className="p-3 bg-gray-50 rounded-lg border">
                    <div className="text-sm">
                      <span className="font-medium">Raw prediction:</span>{" "}
                      <span className="capitalize">
                        {prediction.raw_prediction}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Showing smoothed result above (majority vote from recent
                      predictions)
                    </div>
                  </div>
                )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              {isConnected
                ? "Waiting for IMU data..."
                : "Connect to backend to see predictions"}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Instructions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm text-gray-600">
            <div>
              • The AI model analyzes the last 128 IMU samples (~0.4 seconds)
            </div>
            <div>
              • Predictions are smoothed using a majority vote from recent
              results
            </div>
            <div>• The model was trained on your collected labeled data</div>
            <div>
              • Green = high confidence, Yellow = medium, Red = low confidence
            </div>
            <div>• Buffer must be filled before predictions are available</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
