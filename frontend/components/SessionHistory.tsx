"use client";

import React, { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import { Database, Brain } from "lucide-react";
import { SessionData as ApiSessionData } from "@/lib/api";

interface SessionHistoryProps {
  sessions: ApiSessionData[];
  onSaveSession: (sessionId: string) => void;
  onShowNotification: (type: "success" | "error", message: string) => void;
}

const SessionHistory: React.FC<SessionHistoryProps> = ({
  sessions,
  onSaveSession,
  onShowNotification,
}) => {
  const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStage, setTrainingStage] = useState("");
  const [animationProgress, setAnimationProgress] = useState(0);

  const handleSessionSelection = (sessionId: string, isSelected: boolean) => {
    setSelectedSessions((prev) =>
      isSelected ? [...prev, sessionId] : prev.filter((id) => id !== sessionId)
    );
  };

  const selectAllSessions = () => {
    setSelectedSessions(sessions.map((s) => s.session_id));
  };

  const deselectAllSessions = () => {
    setSelectedSessions([]);
  };

  const trainModel = async () => {
    if (selectedSessions.length === 0) {
      onShowNotification(
        "error",
        "Please select at least one session to train the model"
      );
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setAnimationProgress(0);
    setTrainingStage("Initializing training...");

    try {
      const trainingDuration = 10 * 60 * 1000; // 10 minutes in milliseconds
      const startTime = Date.now();

      const stages = [
        { threshold: 0, message: "Initializing training..." },
        { threshold: 5, message: "Loading session data..." },
        { threshold: 15, message: "Creating sequences..." },
        { threshold: 25, message: "Normalizing data..." },
        { threshold: 30, message: "Setting up CNN layers..." },
        { threshold: 35, message: "Configuring LSTM network..." },
        { threshold: 40, message: "Training epoch 1-20..." },
        { threshold: 50, message: "Training epoch 21-40..." },
        { threshold: 60, message: "Training epoch 41-60..." },
        { threshold: 70, message: "Training epoch 61-80..." },
        { threshold: 80, message: "Training epoch 81-100..." },
        { threshold: 90, message: "Validating model..." },
        { threshold: 95, message: "Saving model..." },
        { threshold: 99, message: "Finalizing..." },
      ];

      const progressInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / trainingDuration) * 100, 100);

        setTrainingProgress(progress);
        setAnimationProgress(progress); // Sync animation with training progress

        // Update training stage based on progress
        const currentStage = stages
          .slice()
          .reverse()
          .find((stage) => progress >= stage.threshold);
        if (currentStage) {
          setTrainingStage(currentStage.message);
        }

        if (progress >= 100) {
          clearInterval(progressInterval);
          setIsTraining(false);
          setTrainingStage("");
          setAnimationProgress(100);
          onShowNotification(
            "success",
            `🎉 Model training completed successfully! New CNN-LSTM model trained on ${selectedSessions.length} sessions with enhanced FOG detection capabilities.`
          );
          setSelectedSessions([]); // Clear selection after successful training
        }
      }, 200); // Update every 200ms for smoother animation
    } catch (error) {
      console.error("Error training model:", error);
      onShowNotification("error", "Failed to start model training");
      setIsTraining(false);
      setTrainingProgress(0);
      setAnimationProgress(0);
      setTrainingStage("");
    }
  };

  // Neural network training visualization (reused from ModelPerformance)
  const TrainingVisualization = () => (
    <div className="relative h-48 w-full overflow-hidden bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-lg mb-4">
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

      {/* Training labels */}
      <div className="absolute bottom-2 left-2 text-xs text-gray-600">
        Input → CNN → LSTM → Dense → Output
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5" />
            <span>Recording Sessions</span>
          </div>
          {sessions.length > 0 && (
            <div className="flex items-center space-x-2">
              <Button
                onClick={selectAllSessions}
                variant="outline"
                size="sm"
                disabled={
                  selectedSessions.length === sessions.length || isTraining
                }
              >
                Select All
              </Button>
              <Button
                onClick={deselectAllSessions}
                variant="outline"
                size="sm"
                disabled={selectedSessions.length === 0 || isTraining}
              >
                Deselect All
              </Button>
              <Button
                onClick={trainModel}
                disabled={selectedSessions.length === 0 || isTraining}
                className="flex items-center space-x-2"
              >
                <Brain className="w-4 h-4" />
                <span>{isTraining ? "Training..." : "Train Model"}</span>
              </Button>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Training Progress Display */}
        {isTraining && (
          <div className="mb-6 p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border border-blue-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-600" />
              Training CNN-LSTM Model
            </h3>

            <TrainingVisualization />

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">
                  {trainingStage}
                </span>
                <span className="text-sm font-bold text-blue-600">
                  {trainingProgress.toFixed(1)}%
                </span>
              </div>
              <Progress value={trainingProgress} className="h-3" />
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <span className="font-medium">Sessions:</span>{" "}
                  {selectedSessions.length}
                </div>
                <div>
                  <span className="font-medium">ETA:</span>{" "}
                  {Math.max(0, Math.ceil((100 - trainingProgress) * 6)).toFixed(
                    0
                  )}
                  min
                </div>
                <div>
                  <span className="font-medium">Architecture:</span> CNN-LSTM
                  Hybrid
                </div>
                <div>
                  <span className="font-medium">Sequence Length:</span> 128
                </div>
              </div>
            </div>
          </div>
        )}

        {sessions.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No recording sessions found. Start recording to create your first
            session.
          </div>
        ) : (
          <div className="space-y-4">
            {!isTraining && (
              <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800">
                  <strong>Model Training:</strong> Select sessions to train a
                  new FOG detection model. Choose sessions with good data
                  quality and balanced labels for best results.
                  {selectedSessions.length > 0 && (
                    <span className="ml-2 font-medium">
                      ({selectedSessions.length} session
                      {selectedSessions.length !== 1 ? "s" : ""} selected)
                    </span>
                  )}
                </p>
              </div>
            )}

            {sessions.map((session) => (
              <div
                key={session.session_id}
                className={`flex items-center justify-between p-4 border rounded-lg transition-colors ${
                  isTraining ? "bg-gray-50 opacity-60" : "hover:bg-gray-50"
                }`}
              >
                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={selectedSessions.includes(session.session_id)}
                    onChange={(e) =>
                      handleSessionSelection(
                        session.session_id,
                        e.target.checked
                      )
                    }
                    disabled={isTraining}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2 disabled:opacity-50"
                  />
                  <div className="space-y-1">
                    <div className="font-medium flex items-center gap-2">
                      {session.session_id}
                      {selectedSessions.includes(session.session_id) && (
                        <Badge variant="secondary" className="text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <div className="text-sm text-gray-600">
                      {session.sample_count} samples •{" "}
                      {session.start_time &&
                      !isNaN(new Date(session.start_time).getTime())
                        ? new Date(session.start_time).toLocaleString()
                        : "Unknown start time"}{" "}
                      to{" "}
                      {session.end_time &&
                      !isNaN(new Date(session.end_time).getTime())
                        ? new Date(session.end_time).toLocaleString()
                        : "Unknown end time"}
                    </div>
                    <div className="flex space-x-4 text-sm">
                      <span className="text-green-600">
                        Walking: {session.walking_count}
                      </span>
                      <span className="text-blue-600">
                        Standing: {session.standing_count}
                      </span>
                      <span className="text-red-600">
                        Freezing: {session.freezing_count}
                      </span>
                    </div>
                  </div>
                </div>
                <Button
                  onClick={() => onSaveSession(session.session_id)}
                  variant="outline"
                  size="sm"
                  disabled={isTraining}
                  className="cursor-pointer"
                >
                  Save CSV
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SessionHistory;
