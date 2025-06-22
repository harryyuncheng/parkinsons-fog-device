"use client";

import { useState, useEffect } from "react";
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
  Legend,
} from "recharts";
import {
  Play,
  Square,
  Wifi,
  WifiOff,
  Database,
  Activity,
  AlertTriangle,
  User,
  UserCheck,
  CheckCircle,
  X,
} from "lucide-react";

interface DataPoint {
  time: string;
  accelX: number;
  accelY: number;
  accelZ: number;
  gyroX: number;
  gyroY: number;
  gyroZ: number;
  state: string;
}

export default function FreezeOfGaitMonitor() {
  const [activeTab, setActiveTab] = useState("recording");
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentState, setCurrentState] = useState("standing");
  const [data, setData] = useState<DataPoint[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ApiSessionData[]>([]);
  const [sampleCount, setSampleCount] = useState(0);
  const [recordingStartTime, setRecordingStartTime] = useState<number>(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [notification, setNotification] = useState<{
    type: "success" | "error";
    message: string;
    show: boolean;
  }>({ type: "success", message: "", show: false });

  // Show notification function
  const showNotification = (type: "success" | "error", message: string) => {
    setNotification({ type, message, show: true });
    // Auto-hide after 5 seconds
    setTimeout(() => {
      setNotification((prev) => ({ ...prev, show: false }));
    }, 5000);
  };

  // Update recording duration
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isRecording && recordingStartTime > 0) {
      interval = setInterval(() => {
        setRecordingDuration(
          Math.floor((Date.now() - recordingStartTime) / 1000)
        );
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording, recordingStartTime]);

  // Backend connection and real-time data handling
  useEffect(() => {
    const initializeConnection = async () => {
      console.log("Initializing backend connection...");
      const isHealthy = await api.checkHealth();
      setIsConnected(isHealthy);
      console.log("Backend health check:", isHealthy);

      if (isHealthy) {
        // Setup socket event listeners
        socketEvents.onConnect(() => {
          console.log("✅ Connected to backend WebSocket");
          setIsConnected(true);
        });

        socketEvents.onDisconnect(() => {
          console.log("❌ Disconnected from backend WebSocket");
          setIsConnected(false);
        });

        socketEvents.onIMUData((data: ApiIMUData) => {
          console.log("📡 Real ESP32 data received:", data);
          console.log("📊 Data details:", {
            acc_x: data.acc_x,
            acc_y: data.acc_y,
            acc_z: data.acc_z,
            gyro_x: data.gyro_x,
            gyro_y: data.gyro_y,
            gyro_z: data.gyro_z,
            current_state: data.current_state,
          });

          // Create data point for visualization
          const newPoint: DataPoint = {
            time: new Date().toLocaleTimeString(),
            accelX: Number(data.acc_x.toFixed(2)),
            accelY: Number(data.acc_y.toFixed(2)),
            accelZ: Number(data.acc_z.toFixed(2)),
            gyroX: Number(data.gyro_x.toFixed(2)),
            gyroY: Number(data.gyro_y.toFixed(2)),
            gyroZ: Number(data.gyro_z.toFixed(2)),
            state: data.current_state || "standing",
          };

          // Update chart data (keep last 50 points)
          setData((prev) => {
            const newData = [...prev, newPoint].slice(-50);
            console.log(
              "📈 Updated chart data, total points:",
              newData.length,
              "latest point:",
              newPoint
            );
            return newData;
          });
          setCurrentState(data.current_state || "standing");
          setSampleCount((prev) => prev + 1);

          // Update AI prediction data if available
          if (data.ai_prediction) {
            setAiPrediction(data.ai_prediction);
          }
        });

        socketEvents.onStateAnnotation((data) => {
          console.log("🏷️ State annotation received:", data.state);
          setCurrentState(data.state);
        });

        socketEvents.onESP32Status((data) => {
          console.log("🔌 ESP32 status:", data);
        });
      }
    };

    initializeConnection();

    return () => {
      disconnectSocket();
    };
  }, []);

  // Keyboard shortcuts for state annotation
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
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isRecording, activeTab]);

  // Load sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      const result = await api.getSessions();
      if (result.status === "success" && result.data) {
        setSessions(result.data);
      }
    };
    loadSessions();
  }, []);

  const startRecording = async () => {
    if (!isConnected) {
      alert("Please ensure backend is connected first!");
      return;
    }

    try {
      const result = await api.startSession();
      if (result.status === "success" && result.data) {
        setIsRecording(true);
        setSessionId(result.data.session_id);
        setRecordingStartTime(Date.now());
        setRecordingDuration(0);
        setSampleCount(0);
        setData([]);
        console.log("🎬 Recording started:", result.data.session_id);
      }
    } catch (error) {
      console.error("Failed to start recording:", error);
      alert("Failed to start recording. Check console for details.");
    }
  };

  const stopRecording = async () => {
    if (!isRecording || !sessionId) return;

    try {
      const result = await api.stopSession();
      if (result.status === "success") {
        const stoppedSessionId = result.data?.session_id || sessionId;

        setIsRecording(false);
        setSessionId(null);
        setRecordingStartTime(0);
        console.log("⏹️ Recording stopped, session:", stoppedSessionId);

        // Automatically save CSV of the recorded session
        console.log("📁 Auto-saving session data...");
        await saveSessionData(stoppedSessionId);

        // Refresh sessions list
        const sessionsResult = await api.getSessions();
        if (sessionsResult.status === "success" && sessionsResult.data) {
          setSessions(sessionsResult.data);
        }
      }
    } catch (error) {
      console.error("Failed to stop recording:", error);
    }
  };

  const annotateCurrentState = async (
    state: "walking" | "standing" | "freezing"
  ) => {
    try {
      const result = await api.annotateState(state);
      if (result.status === "success") {
        console.log("🏷️ State annotated:", state);
        setCurrentState(state);
      }
    } catch (error) {
      console.error("Failed to annotate state:", error);
    }
  };

  const saveSessionData = async (sessionId: string) => {
    try {
      const result = await api.saveSessionCSV(sessionId);
      if (result.status === "success" && result.data) {
        console.log("📁 CSV saved:", result.data.message);
        showNotification(
          "success",
          `Session data saved! ${result.data.samples} samples saved to backend/data/ directory`
        );
      } else {
        console.error("Failed to save session data:", result.error);
        showNotification(
          "error",
          "Failed to save session data. Check console for details."
        );
      }
    } catch (error) {
      console.error("Failed to save session data:", error);
      showNotification(
        "error",
        "Failed to save session data. Check console for details."
      );
    }
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case "walking":
        return "bg-green-100 text-green-800";
      case "freezing":
        return "bg-red-100 text-red-800";
      case "standing":
      default:
        return "bg-blue-100 text-blue-800";
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case "walking":
        return <User className="w-4 h-4" />;
      case "freezing":
        return <AlertTriangle className="w-4 h-4" />;
      case "standing":
      default:
        return <UserCheck className="w-4 h-4" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">
            Parkinson's FOG Detection System
          </h1>
          <div className="flex items-center space-x-4">
            <Badge variant={isConnected ? "default" : "destructive"}>
              {isConnected ? (
                <Wifi className="w-4 h-4 mr-2" />
              ) : (
                <WifiOff className="w-4 h-4 mr-2" />
              )}
              {isConnected ? "Backend Connected" : "Backend Disconnected"}
            </Badge>
            <Badge className={getStateColor(currentState)}>
              {getStateIcon(currentState)}
              <span className="ml-2 capitalize">{currentState}</span>
            </Badge>
          </div>
        </div>

        {/* Connection Status Alert */}
        {!isConnected && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2 text-red-800">
                <AlertTriangle className="w-5 h-5" />
                <span>
                  Backend not connected. Please ensure Flask server is running
                  on port 8080 and ESP32 connector is active.
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recording">Real-time Recording</TabsTrigger>
            <TabsTrigger value="sessions">Session History</TabsTrigger>
          </TabsList>

          <TabsContent value="recording" className="space-y-6">
            {/* Control Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Activity className="w-5 h-5" />
                  <span>ESP32 Data Recording</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex space-x-4">
                    <Button
                      onClick={startRecording}
                      disabled={!isConnected || isRecording}
                      className="flex items-center space-x-2"
                    >
                      <Play className="w-4 h-4" />
                      <span>Start Recording</span>
                    </Button>
                    <Button
                      onClick={stopRecording}
                      disabled={!isRecording}
                      variant="outline"
                      className="flex items-center space-x-2"
                    >
                      <Square className="w-4 h-4" />
                      <span>Stop Recording</span>
                    </Button>
                  </div>
                  {isRecording && sessionId && (
                    <Badge variant="outline">Recording: {sessionId}</Badge>
                  )}
                </div>

                {/* State Annotation Controls */}
                {isRecording && (
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h3 className="text-sm font-medium mb-3">
                      State Annotation
                    </h3>
                    <div className="flex space-x-3">
                      <Button
                        onClick={() => annotateCurrentState("walking")}
                        variant={
                          currentState === "walking" ? "default" : "outline"
                        }
                        size="sm"
                        className="flex items-center space-x-2"
                      >
                        <User className="w-4 h-4" />
                        <span>Walking (W)</span>
                      </Button>
                      <Button
                        onClick={() => annotateCurrentState("standing")}
                        variant={
                          currentState === "standing" ? "default" : "outline"
                        }
                        size="sm"
                        className="flex items-center space-x-2"
                      >
                        <UserCheck className="w-4 h-4" />
                        <span>Standing (S)</span>
                      </Button>
                      <Button
                        onClick={() => annotateCurrentState("freezing")}
                        variant={
                          currentState === "freezing" ? "default" : "outline"
                        }
                        size="sm"
                        className="flex items-center space-x-2"
                      >
                        <AlertTriangle className="w-4 h-4" />
                        <span>Freezing (F)</span>
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Recording Stats */}
            {isRecording && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {sampleCount}
                    </div>
                    <div className="text-sm text-gray-600">ESP32 Samples</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {recordingDuration}s
                    </div>
                    <div className="text-sm text-gray-600">Duration</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4 text-center">
                    <div
                      className={`text-2xl font-bold ${
                        currentState === "freezing"
                          ? "text-red-600"
                          : "text-gray-400"
                      }`}
                    >
                      {currentState === "freezing" ? "FREEZING" : "NORMAL"}
                    </div>
                    <div className="text-sm text-gray-600">FOG Status</div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Real-time Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Accelerometer (ESP32)</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accelX"
                        stroke="#ef4444"
                        name="X-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="accelY"
                        stroke="#22c55e"
                        name="Y-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="accelZ"
                        stroke="#3b82f6"
                        name="Z-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Gyroscope (ESP32)</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="gyroX"
                        stroke="#ef4444"
                        name="X-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="gyroY"
                        stroke="#22c55e"
                        name="Y-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="gyroZ"
                        stroke="#3b82f6"
                        name="Z-axis"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="sessions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="w-5 h-5" />
                  <span>Recording Sessions</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {sessions.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No recording sessions found. Start recording to create your
                    first session.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {sessions.map((session) => (
                      <div
                        key={session.session_id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50"
                      >
                        <div className="space-y-1">
                          <div className="font-medium">
                            {session.session_id}
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
                        <Button
                          onClick={() => saveSessionData(session.session_id)}
                          variant="outline"
                          size="sm"
                        >
                          Save CSV
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Success/Error Notification */}
      {notification.show && (
        <div
          className={`fixed top-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg transition-all duration-300 ${
            notification.type === "success"
              ? "bg-green-50 border border-green-200"
              : "bg-red-50 border border-red-200"
          }`}
        >
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              {notification.type === "success" ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-600" />
              )}
            </div>
            <div className="flex-1">
              <p
                className={`text-sm font-medium ${
                  notification.type === "success"
                    ? "text-green-800"
                    : "text-red-800"
                }`}
              >
                {notification.type === "success" ? "Success!" : "Error"}
              </p>
              <p
                className={`text-sm mt-1 ${
                  notification.type === "success"
                    ? "text-green-700"
                    : "text-red-700"
                }`}
              >
                {notification.message}
              </p>
            </div>
            <button
              onClick={() =>
                setNotification((prev) => ({ ...prev, show: false }))
              }
              className={`flex-shrink-0 p-1 rounded-full hover:bg-opacity-20 ${
                notification.type === "success"
                  ? "hover:bg-green-600"
                  : "hover:bg-red-600"
              }`}
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>
        </div>
      )}

      {/* Instructions */}
      <Card className="fixed bottom-4 right-4 w-80 shadow-lg">
        <CardHeader>
          <CardTitle className="text-sm">Instructions</CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-2">
          <div>• Make sure ESP32 is connected and sending data</div>
          <div>• Backend must be running on port 8080</div>
          <div>• Press W/S/F keys to annotate states while recording</div>
          <div>• All real ESP32 data is saved to database</div>
          <div>• CSV files saved automatically to backend/data/ directory</div>
        </CardContent>
      </Card>
    </div>
  );
}
