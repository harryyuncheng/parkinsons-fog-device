"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  api,
  socketEvents,
  IMUData as ApiIMUData,
  SessionData as ApiSessionData,
  disconnectSocket,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import {
  Play,
  Square,
  Wifi,
  WifiOff,
  Database,
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle,
  User,
  UserCheck,
} from "lucide-react";

interface IMUData {
  timestamp: number;
  accelX: number;
  accelY: number;
  accelZ: number;
  gyroX: number;
  gyroY: number;
  gyroZ: number;
  freezeOfGait: boolean;
  isWalking: boolean;
  prediction?: {
    fog: number;
    walking: number;
    fogConfidence: number;
    walkingConfidence: number;
  };
}

interface DataPoint {
  time: string;
  accelX: number;
  accelY: number;
  accelZ: number;
  gyroX: number;
  gyroY: number;
  gyroZ: number;
  fog: boolean;
  walking: boolean;
  fogPrediction?: number;
  walkingPrediction?: number;
  fogConfidence?: number;
  walkingConfidence?: number;
}

interface PredictionAlert {
  id: string;
  timestamp: number;
  type: "fog" | "walking_change";
  confidence: number;
  value: boolean | number;
  message: string;
}

export default function FreezeOfGaitMonitor() {
  const [activeTab, setActiveTab] = useState("recording");
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isFogActive, setIsFogActive] = useState(false);
  const [isWalking, setIsWalking] = useState(false);
  const [data, setData] = useState<DataPoint[]>([]);
  const [sessionData, setSessionData] = useState<IMUData[]>([]);
  const [predictionAlerts, setPredictionAlerts] = useState<PredictionAlert[]>(
    []
  );
  const [currentPrediction, setCurrentPrediction] = useState<{
    fog: number;
    walking: number;
    fogConfidence: number;
    walkingConfidence: number;
  } | null>(null);

  const [stats, setStats] = useState({
    totalSamples: 0,
    fogEvents: 0,
    walkingTime: 0,
    standingTime: 0,
    recordingDuration: 0,
  });

  const [predictionStats, setPredictionStats] = useState({
    totalPredictions: 0,
    fogDetected: 0,
    walkingDetected: 0,
    averageFogConfidence: 0,
    averageWalkingConfidence: 0,
    sessionDuration: 0,
  });

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const fogStartRef = useRef<number | null>(null);
  const walkingStartRef = useRef<number | null>(null);
  const predictionStartRef = useRef<number>(0);
  const lastWalkingState = useRef<boolean>(false);

  // Real backend connection
  useEffect(() => {
    // Check backend health and setup socket connection
    const initializeConnection = async () => {
      const isHealthy = await api.checkHealth();
      setIsConnected(isHealthy);

      if (isHealthy) {
        // Setup socket event listeners
        socketEvents.onConnect(() => {
          console.log("Connected to backend");
          setIsConnected(true);
        });

        socketEvents.onDisconnect(() => {
          console.log("Disconnected from backend");
          setIsConnected(false);
        });

        socketEvents.onIMUData((data: ApiIMUData) => {
          // Convert API IMU data to local format
          const localData: IMUData = {
            timestamp: Date.now(),
            accelX: data.acc_x,
            accelY: data.acc_y,
            accelZ: data.acc_z,
            gyroX: data.gyro_x,
            gyroY: data.gyro_y,
            gyroZ: data.gyro_z,
            freezeOfGait: data.current_state === "freezing",
            isWalking: data.current_state === "walking",
          };

          // Update session data
          setSessionData((prev) => [...prev, localData]);

          // Update visualization data
          const newPoint: DataPoint = {
            time: new Date(localData.timestamp).toLocaleTimeString(),
            accelX: Number(localData.accelX.toFixed(2)),
            accelY: Number(localData.accelY.toFixed(2)),
            accelZ: Number(localData.accelZ.toFixed(2)),
            gyroX: Number(localData.gyroX.toFixed(2)),
            gyroY: Number(localData.gyroY.toFixed(2)),
            gyroZ: Number(localData.gyroZ.toFixed(2)),
            fog: localData.freezeOfGait,
            walking: localData.isWalking,
          };

          setData((prev) => [...prev, newPoint].slice(-50));

          // Update stats
          if (isRecording) {
            setStats((prev) => ({
              ...prev,
              totalSamples: prev.totalSamples + 1,
              fogEvents: localData.freezeOfGait
                ? prev.fogEvents + 1
                : prev.fogEvents,
              recordingDuration: Math.floor(
                (Date.now() - startTimeRef.current) / 1000
              ),
            }));
          }
        });

        socketEvents.onStateAnnotation((data) => {
          console.log("State annotation received:", data.state);
          // Update UI state based on annotation
          if (data.state === "walking") {
            setIsWalking(true);
            setIsFogActive(false);
          } else if (data.state === "standing") {
            setIsWalking(false);
            setIsFogActive(false);
          } else if (data.state === "freezing") {
            setIsWalking(false);
            setIsFogActive(true);
          }
        });
      }
    };

    initializeConnection();

    return () => {
      disconnectSocket();
    };
  }, []);

  // Simulate ML model prediction
  const predictStates = useCallback(
    (
      imuData: IMUData
    ): {
      fog: number;
      walking: number;
      fogConfidence: number;
      walkingConfidence: number;
    } => {
      const accelMagnitude = Math.sqrt(
        imuData.accelX ** 2 + imuData.accelY ** 2 + imuData.accelZ ** 2
      );
      const gyroMagnitude = Math.sqrt(
        imuData.gyroX ** 2 + imuData.gyroY ** 2 + imuData.gyroZ ** 2
      );

      // Walking prediction: higher acceleration variance suggests walking
      const walkingScore =
        accelMagnitude > 8 && gyroMagnitude > 5
          ? 0.7 + Math.random() * 0.3
          : Math.random() * 0.4;
      const walkingConfidence = Math.min(
        0.95,
        Math.max(0.1, walkingScore + (Math.random() - 0.5) * 0.2)
      );

      // FOG prediction: low acceleration variance + high gyro variance during walking might indicate FOG
      const fogScore =
        imuData.isWalking && gyroMagnitude > 15 && accelMagnitude < 8
          ? 0.8 + Math.random() * 0.2
          : Math.random() * 0.3;
      const fogConfidence = Math.min(
        0.95,
        Math.max(0.1, fogScore + (Math.random() - 0.5) * 0.2)
      );

      return {
        fog: fogScore > 0.6 ? 1 : 0,
        walking: walkingScore > 0.5 ? 1 : 0,
        fogConfidence: fogConfidence,
        walkingConfidence: walkingConfidence,
      };
    },
    []
  );

  // Generate simulated IMU data
  const generateIMUData = useCallback((): IMUData => {
    const now = Date.now();
    const baseFreq = 0.1;
    const walkingMultiplier = isWalking ? 2.0 : 0.5;
    const fogMultiplier = isFogActive ? 1.5 : 1.0;
    const noiseLevel = walkingMultiplier * fogMultiplier;

    const data: IMUData = {
      timestamp: now,
      accelX:
        Math.sin(now * baseFreq * 0.001) * walkingMultiplier +
        (Math.random() - 0.5) * noiseLevel,
      accelY:
        Math.cos(now * baseFreq * 0.001) * walkingMultiplier +
        (Math.random() - 0.5) * noiseLevel,
      accelZ:
        9.8 +
        Math.sin(now * baseFreq * 0.002) * 0.5 +
        (Math.random() - 0.5) * noiseLevel,
      gyroX:
        Math.sin(now * baseFreq * 0.003) * walkingMultiplier * 10 +
        (Math.random() - 0.5) * noiseLevel * 5,
      gyroY:
        Math.cos(now * baseFreq * 0.002) * walkingMultiplier * 8 +
        (Math.random() - 0.5) * noiseLevel * 5,
      gyroZ:
        Math.sin(now * baseFreq * 0.001) * walkingMultiplier * 5 +
        (Math.random() - 0.5) * noiseLevel * 5,
      freezeOfGait: isFogActive,
      isWalking: isWalking,
    };

    // Add prediction if in prediction mode
    if (isPredicting) {
      const prediction = predictStates(data);
      data.prediction = prediction;
    }

    return data;
  }, [isFogActive, isWalking, isPredicting, predictStates]);

  // Handle keyboard events for state labeling
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!event.repeat && isRecording && activeTab === "recording") {
        if (event.code === "KeyW") {
          event.preventDefault();
          annotateCurrentState("walking");
        } else if (event.code === "KeyS") {
          event.preventDefault();
          annotateCurrentState("standing");
        } else if (event.code === "KeyF") {
          event.preventDefault();
          annotateCurrentState("freezing");
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isRecording, activeTab]);

  // Data streaming simulation
  useEffect(() => {
    if (isConnected && (isRecording || isPredicting)) {
      intervalRef.current = setInterval(() => {
        const newData = generateIMUData();

        // Add to session data
        setSessionData((prev) => [...prev, newData]);

        // Update current prediction
        if (isPredicting && newData.prediction) {
          setCurrentPrediction({
            fog: newData.prediction.fog,
            walking: newData.prediction.walking,
            fogConfidence: newData.prediction.fogConfidence,
            walkingConfidence: newData.prediction.walkingConfidence,
          });

          // Add prediction alerts
          if (
            newData.prediction.fog === 1 &&
            newData.prediction.fogConfidence > 0.7
          ) {
            setPredictionAlerts((prev) => {
              const newAlert: PredictionAlert = {
                id: `fog_alert_${Date.now()}`,
                timestamp: newData.timestamp,
                type: "fog",
                confidence: newData.prediction!.fogConfidence,
                value: true,
                message: "Freeze of Gait detected",
              };
              return [newAlert, ...prev.slice(0, 9)];
            });
          }

          // Alert for walking state changes
          if (
            (newData.prediction.walking === 1) !== lastWalkingState.current &&
            newData.prediction.walkingConfidence > 0.8
          ) {
            setPredictionAlerts((prev) => {
              const newAlert: PredictionAlert = {
                id: `walking_alert_${Date.now()}`,
                timestamp: newData.timestamp,
                type: "walking_change",
                confidence: newData.prediction!.walkingConfidence,
                value: newData.prediction!.walking,
                message:
                  newData.prediction!.walking === 1
                    ? "Walking detected"
                    : "Standing detected",
              };
              return [newAlert, ...prev.slice(0, 9)];
            });
            lastWalkingState.current = newData.prediction.walking === 1;
          }

          // Update prediction stats
          setPredictionStats((prev) => ({
            totalPredictions: prev.totalPredictions + 1,
            fogDetected:
              prev.fogDetected + (newData.prediction!.fog === 1 ? 1 : 0),
            walkingDetected:
              prev.walkingDetected +
              (newData.prediction!.walking === 1 ? 1 : 0),
            averageFogConfidence:
              (prev.averageFogConfidence * prev.totalPredictions +
                newData.prediction!.fogConfidence) /
              (prev.totalPredictions + 1),
            averageWalkingConfidence:
              (prev.averageWalkingConfidence * prev.totalPredictions +
                newData.prediction!.walkingConfidence) /
              (prev.totalPredictions + 1),
            sessionDuration: Math.floor(
              (Date.now() - predictionStartRef.current) / 1000
            ),
          }));
        }

        // Update visualization data (keep last 50 points)
        setData((prev) => {
          const newPoint: DataPoint = {
            time: new Date(newData.timestamp).toLocaleTimeString(),
            accelX: Number(newData.accelX.toFixed(2)),
            accelY: Number(newData.accelY.toFixed(2)),
            accelZ: Number(newData.accelZ.toFixed(2)),
            gyroX: Number(newData.gyroX.toFixed(2)),
            gyroY: Number(newData.gyroY.toFixed(2)),
            gyroZ: Number(newData.gyroZ.toFixed(2)),
            fog: newData.freezeOfGait,
            walking: newData.isWalking,
            fogPrediction: newData.prediction?.fog,
            walkingPrediction: newData.prediction?.walking,
            fogConfidence: newData.prediction?.fogConfidence,
            walkingConfidence: newData.prediction?.walkingConfidence,
          };

          const updated = [...prev, newPoint].slice(-50);
          return updated;
        });

        // Update recording stats
        if (isRecording) {
          setStats((prev) => {
            const currentTime = Date.now();
            const timeDelta = 100; // 100ms interval

            return {
              ...prev,
              totalSamples: prev.totalSamples + 1,
              recordingDuration: Math.floor(
                (currentTime - startTimeRef.current) / 1000
              ),
              standingTime: !isWalking
                ? prev.standingTime + timeDelta
                : prev.standingTime,
            };
          });
        }
      }, 100); // 10Hz sampling rate

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [isConnected, isRecording, isPredicting, generateIMUData, isWalking]);

  const startRecording = async () => {
    try {
      const result = await api.startSession();
      if (result.status === "success") {
        setIsRecording(true);
        startTimeRef.current = Date.now();
        setSessionData([]);
        setData([]);
        setStats({
          totalSamples: 0,
          fogEvents: 0,
          walkingTime: 0,
          standingTime: 0,
          recordingDuration: 0,
        });
        console.log("Recording started:", result.data?.session_id);
      } else {
        console.error("Failed to start recording:", result.error);
      }
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const stopRecording = async () => {
    try {
      const result = await api.stopSession();
      if (result.status === "success") {
        setIsRecording(false);
        setIsFogActive(false);
        setIsWalking(false);

        // Add final walking time if currently walking
        if (walkingStartRef.current) {
          const finalWalkingTime = Date.now() - walkingStartRef.current;
          setStats((prev) => ({
            ...prev,
            walkingTime: prev.walkingTime + finalWalkingTime,
          }));
        }

        console.log("Recording stopped successfully");
      } else {
        console.error("Failed to stop recording:", result.error);
      }
    } catch (error) {
      console.error("Failed to stop recording:", error);
    }
  };

  const startPrediction = () => {
    setIsPredicting(true);
    predictionStartRef.current = Date.now();
    setSessionData([]);
    setData([]);
    setPredictionAlerts([]);
    setCurrentPrediction(null);
    lastWalkingState.current = false;
    setPredictionStats({
      totalPredictions: 0,
      fogDetected: 0,
      walkingDetected: 0,
      averageFogConfidence: 0,
      averageWalkingConfidence: 0,
      sessionDuration: 0,
    });
  };

  const stopPrediction = async () => {
    setIsPredicting(false);
    console.log("Prediction stopped");
  };

  // New function to handle state annotations
  const annotateCurrentState = async (
    state: "walking" | "standing" | "freezing"
  ) => {
    if (!isRecording) {
      console.warn("Recording not active, annotation ignored");
      return;
    }

    try {
      const result = await api.annotateState(state);
      if (result.status === "success") {
        // Update local UI state
        if (state === "walking") {
          setIsWalking(true);
          setIsFogActive(false);
        } else if (state === "standing") {
          setIsWalking(false);
          setIsFogActive(false);
        } else if (state === "freezing") {
          setIsWalking(false);
          setIsFogActive(true);
        }
        console.log(`State annotated as: ${state}`);
      } else {
        console.error("Failed to annotate state:", result.error);
      }
    } catch (error) {
      console.error("Error annotating state:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Freeze of Gait Monitor
            </h1>
            <p className="text-gray-600">
              Real-time IMU data analysis for Parkinson's patients
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Badge
              variant={isConnected ? "default" : "secondary"}
              className="flex items-center gap-2"
            >
              {isConnected ? (
                <Wifi className="w-4 h-4" />
              ) : (
                <WifiOff className="w-4 h-4" />
              )}
              {isConnected ? "ESP32 Connected" : "Connecting..."}
            </Badge>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recording" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Data Recording
            </TabsTrigger>
            <TabsTrigger value="prediction" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Live Prediction
            </TabsTrigger>
          </TabsList>

          {/* Recording Tab */}
          <TabsContent value="recording" className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                Training Data Collection
              </h2>
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={!isConnected}
                variant={isRecording ? "destructive" : "default"}
                className="flex items-center gap-2"
              >
                {isRecording ? (
                  <Square className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isRecording ? "Stop Recording" : "Start Recording"}
              </Button>
            </div>

            {/* Recording Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Recording Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        isRecording ? "bg-red-500 animate-pulse" : "bg-gray-300"
                      }`}
                    />
                    <span className="text-lg font-semibold">
                      {isRecording ? "Recording" : "Stopped"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Movement State
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {isWalking ? (
                      <UserCheck className="w-4 h-4 text-blue-500" />
                    ) : (
                      <User className="w-4 h-4 text-gray-500" />
                    )}
                    <span className="text-lg font-semibold">
                      {isWalking ? "Walking" : "Standing"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Freeze of Gait
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        isFogActive
                          ? "bg-orange-500 animate-pulse"
                          : "bg-green-500"
                      }`}
                    />
                    <span className="text-lg font-semibold">
                      {isFogActive ? "Active" : "Normal"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Total Samples
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-blue-500" />
                    <span className="text-lg font-semibold">
                      {stats.totalSamples}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    FOG Events
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-orange-500" />
                    <span className="text-lg font-semibold">
                      {stats.fogEvents}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Instructions */}
            <Card className="bg-blue-50 border-blue-200">
              <CardContent className="pt-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="text-blue-600">
                      <kbd className="px-2 py-1 bg-blue-100 rounded text-sm font-mono">
                        W
                      </kbd>
                    </div>
                    <div>
                      <p className="font-medium text-blue-900">
                        Press W to label current state as Walking
                      </p>
                      <p className="text-sm text-blue-700">
                        Patient is actively walking
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-blue-600">
                      <kbd className="px-2 py-1 bg-blue-100 rounded text-sm font-mono">
                        S
                      </kbd>
                    </div>
                    <div>
                      <p className="font-medium text-blue-900">
                        Press S to label current state as Standing
                      </p>
                      <p className="text-sm text-blue-700">
                        Patient is standing still
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-blue-600">
                      <kbd className="px-2 py-1 bg-blue-100 rounded text-sm font-mono">
                        F
                      </kbd>
                    </div>
                    <div>
                      <p className="font-medium text-blue-900">
                        Press F to label current state as Freezing
                      </p>
                      <p className="text-sm text-blue-700">
                        Patient is experiencing freezing of gait
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Prediction Tab */}
          <TabsContent value="prediction" className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">
                Real-time State Detection
              </h2>
              <Button
                onClick={isPredicting ? stopPrediction : startPrediction}
                disabled={!isConnected}
                variant={isPredicting ? "destructive" : "default"}
                className="flex items-center gap-2"
              >
                {isPredicting ? (
                  <Square className="w-4 h-4" />
                ) : (
                  <Brain className="w-4 h-4" />
                )}
                {isPredicting ? "Stop Monitoring" : "Start Monitoring"}
              </Button>
            </div>

            {/* Prediction Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Monitoring Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        isPredicting
                          ? "bg-green-500 animate-pulse"
                          : "bg-gray-300"
                      }`}
                    />
                    <span className="text-lg font-semibold">
                      {isPredicting ? "Active" : "Stopped"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Movement Prediction
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {currentPrediction?.walking === 1 ? (
                      <UserCheck className="w-4 h-4 text-blue-500" />
                    ) : (
                      <User className="w-4 h-4 text-gray-500" />
                    )}
                    <span className="text-lg font-semibold">
                      {currentPrediction?.walking === 1
                        ? "Walking"
                        : "Standing"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    FOG Prediction
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {currentPrediction?.fog === 1 ? (
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    )}
                    <span className="text-lg font-semibold">
                      {currentPrediction?.fog === 1 ? "FOG Detected" : "Normal"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    FOG Confidence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-purple-500" />
                    <span className="text-lg font-semibold">
                      {currentPrediction
                        ? `${(currentPrediction.fogConfidence * 100).toFixed(
                            1
                          )}%`
                        : "N/A"}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Walking Confidence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-blue-500" />
                    <span className="text-lg font-semibold">
                      {currentPrediction
                        ? `${(
                            currentPrediction.walkingConfidence * 100
                          ).toFixed(1)}%`
                        : "N/A"}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recent Alerts */}
            {predictionAlerts.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Detections</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {predictionAlerts.map((alert) => (
                      <div
                        key={alert.id}
                        className={`flex items-center justify-between p-2 rounded border-l-4 ${
                          alert.type === "fog"
                            ? "bg-red-50 border-red-400"
                            : "bg-blue-50 border-blue-400"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {alert.type === "fog" ? (
                            <AlertTriangle className="w-4 h-4 text-red-500" />
                          ) : (
                            <UserCheck className="w-4 h-4 text-blue-500" />
                          )}
                          <span className="text-sm font-medium">
                            {alert.message}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600">
                          {new Date(alert.timestamp).toLocaleTimeString()} -{" "}
                          {(alert.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Charts - Common for both tabs */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Accelerometer Data */}
          <Card>
            <CardHeader>
              <CardTitle>Accelerometer Data (m/s²)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="accelX"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Accel X"
                    />
                    <Line
                      type="monotone"
                      dataKey="accelY"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={false}
                      name="Accel Y"
                    />
                    <Line
                      type="monotone"
                      dataKey="accelZ"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Accel Z"
                    />
                    {/* Walking periods background */}
                    {data.some((d) => d.walking) && (
                      <ReferenceLine
                        y={0}
                        stroke="#3b82f6"
                        strokeDasharray="2 2"
                        strokeOpacity={0.3}
                      />
                    )}
                    {/* FOG events */}
                    {data.some((d) => d.fog || d.fogPrediction === 1) && (
                      <ReferenceLine
                        y={0}
                        stroke="#f59e0b"
                        strokeDasharray="5 5"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Gyroscope Data */}
          <Card>
            <CardHeader>
              <CardTitle>Gyroscope Data (°/s)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="gyroX"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Gyro X"
                    />
                    <Line
                      type="monotone"
                      dataKey="gyroY"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot={false}
                      name="Gyro Y"
                    />
                    <Line
                      type="monotone"
                      dataKey="gyroZ"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Gyro Z"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Session Info */}
        <Card>
          <CardHeader>
            <CardTitle>Session Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div>
                <p className="text-sm text-gray-600">Session Duration</p>
                <p className="text-lg font-semibold">
                  {activeTab === "recording"
                    ? `${Math.floor(stats.recordingDuration / 60)}:${(
                        stats.recordingDuration % 60
                      )
                        .toString()
                        .padStart(2, "0")}`
                    : `${Math.floor(predictionStats.sessionDuration / 60)}:${(
                        predictionStats.sessionDuration % 60
                      )
                        .toString()
                        .padStart(2, "0")}`}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Data Points</p>
                <p className="text-lg font-semibold">{sessionData.length}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">
                  {activeTab === "recording"
                    ? "Walking Time"
                    : "Walking Detected"}
                </p>
                <p className="text-lg font-semibold">
                  {activeTab === "recording"
                    ? `${Math.floor(stats.walkingTime / 60000)}:${Math.floor(
                        (stats.walkingTime % 60000) / 1000
                      )
                        .toString()
                        .padStart(2, "0")}`
                    : predictionStats.walkingDetected}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">
                  {activeTab === "recording" ? "Standing Time" : "FOG Detected"}
                </p>
                <p className="text-lg font-semibold">
                  {activeTab === "recording"
                    ? `${Math.floor(stats.standingTime / 60000)}:${Math.floor(
                        (stats.standingTime % 60000) / 1000
                      )
                        .toString()
                        .padStart(2, "0")}`
                    : predictionStats.fogDetected}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">
                  {activeTab === "recording" ? "FOG Events" : "Avg Confidence"}
                </p>
                <p className="text-lg font-semibold">
                  {activeTab === "recording"
                    ? stats.fogEvents
                    : predictionStats.averageFogConfidence > 0
                    ? `${(predictionStats.averageFogConfidence * 100).toFixed(
                        1
                      )}%`
                    : "N/A"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
