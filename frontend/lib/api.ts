import io from "socket.io-client";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8080";

export interface IMUData {
  timestamp: string;
  acc_x: number;
  acc_y: number;
  acc_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
  current_state?: string;
}

export interface SessionData {
  session_id: string;
  sample_count: number;
  start_time: string;
  end_time: string;
  walking_count: number;
  standing_count: number;
  freezing_count: number;
}

export interface ApiResponse<T = any> {
  status: string;
  data?: T;
  message?: string;
  error?: string;
}

// Socket.IO connection
let socket: any = null;

export const connectSocket = () => {
  if (!socket) {
    socket = io(BACKEND_URL, {
      transports: ["websocket", "polling"],
    });
  }
  return socket;
};

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};

// API functions
export const api = {
  // Session management
  async startSession(): Promise<ApiResponse<{ session_id: string }>> {
    try {
      const response = await fetch(`${BACKEND_URL}/start_session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },

  async stopSession(): Promise<ApiResponse<{ session_id: string }>> {
    try {
      const response = await fetch(`${BACKEND_URL}/stop_session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },

  // State annotation
  async annotateState(
    state: "walking" | "standing" | "freezing"
  ): Promise<ApiResponse> {
    try {
      const response = await fetch(`${BACKEND_URL}/annotate_state`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ state }),
      });
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },

  // Data retrieval
  async getSessions(): Promise<ApiResponse<SessionData[]>> {
    try {
      const response = await fetch(`${BACKEND_URL}/get_sessions`);
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },

  async getSessionData(sessionId: string): Promise<ApiResponse<any[]>> {
    try {
      const response = await fetch(
        `${BACKEND_URL}/get_session_data/${sessionId}`
      );
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },

  // Health check
  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${BACKEND_URL}/`);
      return response.ok;
    } catch (error) {
      return false;
    }
  },

  // Save session as CSV in backend
  async saveSessionCSV(
    sessionId: string
  ): Promise<
    ApiResponse<{ message: string; filepath: string; samples: number }>
  > {
    try {
      const response = await fetch(
        `${BACKEND_URL}/save_session_csv/${sessionId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = await response.json();
      return { status: "success", data };
    } catch (error) {
      return {
        status: "error",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  },
};

// Socket event handlers
export const socketEvents = {
  onConnect: (callback: () => void) => {
    const socket = connectSocket();
    socket.on("connect", callback);
  },

  onDisconnect: (callback: () => void) => {
    const socket = connectSocket();
    socket.on("disconnect", callback);
  },

  onIMUData: (callback: (data: IMUData) => void) => {
    const socket = connectSocket();
    socket.on("imu_data", callback);
  },

  onStateAnnotation: (
    callback: (data: { state: string; timestamp: string }) => void
  ) => {
    const socket = connectSocket();
    socket.on("state_annotation", callback);
  },

  onStatus: (callback: (data: { message: string }) => void) => {
    const socket = connectSocket();
    socket.on("status", callback);
  },

  onESP32Status: (callback: (data: any) => void) => {
    const socket = connectSocket();
    socket.on("esp32_status", callback);
  },

  // Remove specific event listeners
  off: (event: string, callback?: Function) => {
    if (socket) {
      socket.off(event, callback);
    }
  },
};

// Utility functions
export const downloadSessionCSV = async (sessionId: string) => {
  try {
    const result = await api.getSessionData(sessionId);
    if (result.status === "success" && result.data) {
      const csvContent = convertToCSV(result.data);
      downloadCSV(csvContent, `session_${sessionId}.csv`);
    }
  } catch (error) {
    console.error("Error downloading session data:", error);
  }
};

const convertToCSV = (data: any[]): string => {
  if (!data.length) return "";

  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(","),
    ...data.map((row) => headers.map((header) => row[header]).join(",")),
  ].join("\n");

  return csvContent;
};

const downloadCSV = (content: string, filename: string) => {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.setAttribute("href", url);
  link.setAttribute("download", filename);
  link.style.visibility = "hidden";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export default api;
