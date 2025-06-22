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
  Brain,
} from "lucide-react";
import AIMonitoring from "@/components/AIMonitoring";

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
  const [aiPrediction, setAiPrediction] = useState<PredictionData | undefined>(
    undefined
  );

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
            setAiPrediction(data.ai_prediction as PredictionData);
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
    <div className="min-h-screen bg-stone-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold text-stone-100">
              Parkinson&apos;s FOG Detection
            </h1>
            <p className="text-stone-400">
              Real-time monitoring and analysis system
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="modern-card rounded-lg px-4 py-2 flex items-center space-x-2 status-connected">
              <div
                className={`status-dot ${
                  isConnected ? "connected" : "disconnected"
                }`}
              ></div>
              {isConnected ? (
                <Wifi className="w-4 h-4 text-amber-400" />
              ) : (
                <WifiOff className="w-4 h-4 text-stone-500" />
              )}
              <span
                className={`text-sm font-medium ${
                  isConnected ? "text-amber-200" : "text-stone-400"
                }`}
              >
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
            <div
              className={`modern-card rounded-lg px-4 py-2 flex items-center space-x-2 ${
                currentState === "freezing"
                  ? "state-freezing"
                  : currentState === "walking"
                  ? "state-walking"
                  : "state-standing"
              }`}
            >
              {getStateIcon(currentState)}
              <span className="font-medium capitalize">{currentState}</span>
            </div>
          </div>
        </div>

        {/* Connection Status Alert */}
        {!isConnected && (
          <div className="modern-card border-orange-500/40 notification-error rounded-lg">
            <div className="p-4">
              <div className="flex items-center space-x-3 text-orange-200">
                <AlertTriangle className="w-5 h-5" />
                <div>
                  <h3 className="font-semibold">Backend Connection Required</h3>
                  <p className="text-sm text-orange-300 mt-1">
                    Please ensure Flask server is running on port 6000 and ESP32
                    connector is active.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="modern-card rounded-lg overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <div className="bg-stone-800/50 p-1">
              <TabsList className="grid w-full grid-cols-3 bg-stone-900 rounded-md">
                <TabsTrigger
                  value="recording"
                  className="tab-active data-[state=active]:bg-gradient-to-r data-[state=active]:from-amber-500 data-[state=active]:to-amber-600 data-[state=active]:text-black text-stone-300"
                >
                  Real-time Recording
                </TabsTrigger>
                <TabsTrigger
                  value="monitoring"
                  className="tab-active data-[state=active]:bg-gradient-to-r data-[state=active]:from-amber-500 data-[state=active]:to-amber-600 data-[state=active]:text-black text-stone-300"
                >
                  AI Monitor
                </TabsTrigger>
                <TabsTrigger
                  value="sessions"
                  className="tab-active data-[state=active]:bg-gradient-to-r data-[state=active]:from-amber-500 data-[state=active]:to-amber-600 data-[state=active]:text-black text-stone-300"
                >
                  Session History
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="recording" className="p-6 space-y-6">
              {/* Control Panel */}
              <div className="modern-card rounded-lg overflow-hidden">
                <div className="section-header-primary p-4">
                  <h2 className="text-xl font-bold flex items-center space-x-2">
                    <Activity className="w-5 h-5" />
                    <span>ESP32 Data Recording</span>
                  </h2>
                </div>
                <div className="p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex space-x-3">
                      <Button
                        onClick={startRecording}
                        disabled={!isConnected || isRecording}
                        className={`section-header-primary hover:opacity-90 text-black px-4 py-2 rounded-md font-medium transition-all ${
                          isRecording ? "opacity-50" : ""
                        }`}
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Start Recording
                      </Button>
                      <Button
                        onClick={stopRecording}
                        disabled={!isRecording}
                        variant="outline"
                        className={`border-stone-500/40 text-stone-200 hover:bg-stone-700/30 px-4 py-2 rounded-md font-medium transition-all ${
                          !isRecording ? "opacity-50" : ""
                        }`}
                      >
                        <Square className="w-4 h-4 mr-2" />
                        Stop Recording
                      </Button>
                    </div>
                    {isRecording && sessionId && (
                      <div className="pulse-recording text-white px-3 py-1 rounded-md text-sm font-medium">
                        ● REC {sessionId}
                      </div>
                    )}
                  </div>

                  {/* State Annotation Controls */}
                  {isRecording && (
                    <div className="modern-card p-4 rounded-lg bg-gray-50">
                      <h3 className="text-sm font-semibold mb-3 text-gray-800">
                        State Annotation
                      </h3>
                      <div className="flex gap-2 mb-4">
                        <button
                          onClick={() => annotateCurrentState("walking")}
                          className={`px-4 py-2 rounded-lg border transition-colors btn-walking ${
                            currentState === "walking" ? "active" : ""
                          }`}
                        >
                          Walking (W)
                        </button>
                        <button
                          onClick={() => annotateCurrentState("standing")}
                          className={`px-4 py-2 rounded-lg border transition-colors btn-standing ${
                            currentState === "standing" ? "active" : ""
                          }`}
                        >
                          Standing (S)
                        </button>
                        <button
                          onClick={() => annotateCurrentState("freezing")}
                          className={`px-4 py-2 rounded-lg border transition-colors btn-freezing ${
                            currentState === "freezing" ? "active" : ""
                          }`}
                        >
                          Freezing (F)
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Recording Stats */}
              {isRecording && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="modern-card rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-amber-400 mb-1">
                      {sampleCount.toLocaleString()}
                    </div>
                    <div className="text-sm text-stone-400">ESP32 Samples</div>
                  </div>
                  <div className="modern-card rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-amber-400 mb-1">
                      {Math.floor(recordingDuration / 60)}:
                      {(recordingDuration % 60).toString().padStart(2, "0")}
                    </div>
                    <div className="text-sm text-stone-400">Duration</div>
                  </div>
                  <div className="modern-card rounded-lg p-4 text-center">
                    <div
                      className={`text-2xl font-bold mb-1 ${
                        currentState === "freezing"
                          ? "text-orange-400"
                          : "text-stone-400"
                      }`}
                    >
                      {currentState === "freezing" ? "ALERT" : "NORMAL"}
                    </div>
                    <div className="text-sm text-stone-400">FOG Status</div>
                  </div>
                </div>
              )}

              {/* Real-time Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="modern-card rounded-lg overflow-hidden">
                  <div className="section-header-primary p-4">
                    <h3 className="text-lg font-semibold">
                      Accelerometer Data
                    </h3>
                    <p className="text-black/70 text-sm">
                      Real-time ESP32 acceleration
                    </p>
                  </div>
                  <div className="p-4">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#44403c" />
                        <XAxis dataKey="time" stroke="#a8a29e" fontSize={12} />
                        <YAxis stroke="#a8a29e" fontSize={12} />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="accelX"
                          stroke="#d4a574"
                          name="X-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                        <Line
                          type="monotone"
                          dataKey="accelY"
                          stroke="#b45309"
                          name="Y-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                        <Line
                          type="monotone"
                          dataKey="accelZ"
                          stroke="#ea580c"
                          name="Z-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="modern-card rounded-lg overflow-hidden">
                  <div className="section-header-secondary p-4">
                    <h3 className="text-lg font-semibold text-white">
                      Gyroscope Data
                    </h3>
                    <p className="text-white/80 text-sm">
                      Real-time ESP32 rotation
                    </p>
                  </div>
                  <div className="p-4">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#44403c" />
                        <XAxis dataKey="time" stroke="#a8a29e" fontSize={12} />
                        <YAxis stroke="#a8a29e" fontSize={12} />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="gyroX"
                          stroke="#d4a574"
                          name="X-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                        <Line
                          type="monotone"
                          dataKey="gyroY"
                          stroke="#b45309"
                          name="Y-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                        <Line
                          type="monotone"
                          dataKey="gyroZ"
                          stroke="#ea580c"
                          name="Z-axis"
                          dot={false}
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="monitoring" className="p-6">
              <AIMonitoring
                isConnected={isConnected}
                realTimePrediction={aiPrediction}
              />
            </TabsContent>

            <TabsContent value="sessions" className="p-6 space-y-4">
              <div className="modern-card rounded-lg overflow-hidden">
                <div className="section-header-tertiary p-4">
                  <h2 className="text-xl font-bold text-white flex items-center space-x-2">
                    <Database className="w-5 h-5" />
                    <span>Recording Sessions</span>
                  </h2>
                </div>
                <div className="p-6">
                  {sessions.length === 0 ? (
                    <div className="text-center py-8">
                      <Database className="w-12 h-12 text-stone-500 mx-auto mb-3" />
                      <p className="text-stone-400">
                        No recording sessions found
                      </p>
                      <p className="text-stone-500 text-sm mt-1">
                        Start recording to create your first session
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {sessions.map((session) => (
                        <div
                          key={session.session_id}
                          className="modern-card rounded-lg p-4 card-hover"
                        >
                          <div className="flex items-center justify-between">
                            <div className="space-y-2">
                              <div className="font-semibold text-stone-200">
                                {session.session_id}
                              </div>
                              <div className="text-sm text-stone-400">
                                <span className="font-medium">
                                  {session.sample_count.toLocaleString()}
                                </span>{" "}
                                samples •{" "}
                                {session.start_time &&
                                !isNaN(new Date(session.start_time).getTime())
                                  ? new Date(
                                      session.start_time
                                    ).toLocaleString()
                                  : "Unknown start time"}{" "}
                                to{" "}
                                {session.end_time &&
                                !isNaN(new Date(session.end_time).getTime())
                                  ? new Date(session.end_time).toLocaleString()
                                  : "Unknown end time"}
                              </div>
                              <div className="flex space-x-4 text-sm">
                                <span className="text-amber-400 font-medium">
                                  Walking: {session.walking_count}
                                </span>
                                <span className="text-stone-400 font-medium">
                                  Standing: {session.standing_count}
                                </span>
                                <span className="text-orange-400 font-medium">
                                  Freezing: {session.freezing_count}
                                </span>
                              </div>
                            </div>
                            <Button
                              onClick={() =>
                                saveSessionData(session.session_id)
                              }
                              className="section-header-primary hover:opacity-90 text-black px-4 py-2 rounded-md font-medium transition-all"
                            >
                              Save CSV
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Success/Error Notification */}
      {notification.show && (
        <div
          className={`fixed top-4 right-4 z-50 max-w-md modern-card rounded-lg shadow-lg transition-all duration-300 ${
            notification.type === "success"
              ? "notification-success"
              : "notification-error"
          }`}
        >
          <div className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {notification.type === "success" ? (
                  <CheckCircle className="w-5 h-5 text-amber-400" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-orange-400" />
                )}
              </div>
              <div className="flex-1">
                <p
                  className={`font-medium ${
                    notification.type === "success"
                      ? "text-amber-200"
                      : "text-orange-200"
                  }`}
                >
                  {notification.type === "success" ? "Success!" : "Error"}
                </p>
                <p
                  className={`text-sm mt-1 ${
                    notification.type === "success"
                      ? "text-amber-300"
                      : "text-orange-300"
                  }`}
                >
                  {notification.message}
                </p>
              </div>
              <button
                onClick={() =>
                  setNotification((prev) => ({ ...prev, show: false }))
                }
                className="flex-shrink-0 p-1 rounded-full hover:bg-black/20 transition-colors"
              >
                <X className="w-4 h-4 text-stone-500" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Instructions Panel */}
      <div className="fixed bottom-4 right-4 w-72 modern-card rounded-lg shadow-lg">
        <div className="bg-gray-700 p-3 rounded-t-lg">
          <h3 className="font-medium text-white flex items-center space-x-2">
            <Brain className="w-4 h-4" />
            <span>Quick Guide</span>
          </h3>
        </div>
        <div className="p-3 space-y-2 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
            <span>Connect ESP32 and start backend server</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>Use W/S/F keys for state annotation</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-purple-500"></div>
            <span>Data auto-saved to backend/data/</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-orange-500"></div>
            <span>Monitor real-time FOG detection</span>
          </div>
        </div>
      </div>
    </div>
  );
}
