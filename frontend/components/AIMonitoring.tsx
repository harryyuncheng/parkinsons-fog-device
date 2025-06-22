import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Brain,
  RotateCcw,
  Activity,
  CheckCircle,
  AlertTriangle,
  User,
} from "lucide-react";
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

interface BufferStatus {
  current_size: number;
  required_size: number;
  fill_percentage: number;
  ready_for_prediction: boolean;
}

interface AIMonitoringProps {
  isConnected: boolean;
  realTimePrediction?: PredictionData;
}

export default function AIMonitoring({
  isConnected,
  realTimePrediction,
}: AIMonitoringProps) {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [bufferStatus, setBufferStatus] = useState<BufferStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Use real-time prediction if provided, otherwise poll manually
  useEffect(() => {
    if (realTimePrediction) {
      // Only update if the prediction actually changed
      setPrediction((prev) => {
        if (
          !prev ||
          prev.prediction !== realTimePrediction.prediction ||
          prev.confidence !== realTimePrediction.confidence ||
          prev.status !== realTimePrediction.status
        ) {
          setLastUpdate(new Date());
          return realTimePrediction;
        }
        return prev;
      });
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
            setBufferStatus(result.data.buffer_status);
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
        setBufferStatus(null);
      }
    } catch (error) {
      console.error("Error resetting buffer:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case "walking":
        return "bg-green-100 text-green-800 border-green-200";
      case "freezing":
        return "bg-red-100 text-red-800 border-red-200";
      case "standing":
      default:
        return "bg-blue-100 text-blue-800 border-blue-200";
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case "walking":
        return <User className="w-5 h-5" />;
      case "freezing":
        return <AlertTriangle className="w-5 h-5" />;
      case "standing":
      default:
        return <Activity className="w-5 h-5" />;
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
          <CardContent className="p-4">
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

      {/* Buffer Status */}
      {bufferStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Data Buffer Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Buffer Fill</span>
                  <span>
                    {bufferStatus.current_size}/{bufferStatus.required_size}{" "}
                    samples
                  </span>
                </div>
                <Progress
                  value={bufferStatus.fill_percentage}
                  className="h-2"
                />
              </div>
              <div className="flex items-center space-x-2">
                {bufferStatus.ready_for_prediction ? (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                ) : (
                  <AlertTriangle className="w-4 h-4 text-yellow-600" />
                )}
                <span className="text-sm">
                  {bufferStatus.ready_for_prediction
                    ? "Ready for prediction"
                    : "Collecting data..."}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Prediction */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Current AI Prediction</CardTitle>
        </CardHeader>
        <CardContent>
          {prediction ? (
            <div className="space-y-6">
              {/* Main prediction */}
              <div className="text-center">
                <Badge
                  className={`${getStateColor(
                    prediction.prediction
                  )} text-lg px-4 py-2 flex items-center space-x-2 justify-center w-fit mx-auto`}
                >
                  {getStateIcon(prediction.prediction)}
                  <span className="capitalize font-bold">
                    {prediction.prediction}
                  </span>
                </Badge>
                <div
                  className={`text-sm mt-2 font-medium ${getConfidenceColor(
                    prediction.confidence
                  )}`}
                >
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
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
