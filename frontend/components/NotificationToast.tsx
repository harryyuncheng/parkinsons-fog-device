import { CheckCircle, AlertTriangle, X } from "lucide-react";

interface NotificationToastProps {
  type: "success" | "error";
  message: string;
  show: boolean;
  onClose: () => void;
}

export default function NotificationToast({
  type,
  message,
  show,
  onClose,
}: NotificationToastProps) {
  if (!show) return null;

  return (
    <div
      className={`fixed top-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg transition-all duration-300 ${
        type === "success"
          ? "bg-green-50 border border-green-200"
          : "bg-red-50 border border-red-200"
      }`}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {type === "success" ? (
            <CheckCircle className="w-5 h-5 text-green-600" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-red-600" />
          )}
        </div>
        <div className="flex-1">
          <p
            className={`text-sm font-medium ${
              type === "success" ? "text-green-800" : "text-red-800"
            }`}
          >
            {type === "success" ? "Success!" : "Error"}
          </p>
          <p
            className={`text-sm mt-1 ${
              type === "success" ? "text-green-700" : "text-red-700"
            }`}
          >
            {message}
          </p>
        </div>
        <button
          onClick={onClose}
          className={`flex-shrink-0 p-1 rounded-full hover:bg-opacity-20 ${
            type === "success" ? "hover:bg-green-600" : "hover:bg-red-600"
          }`}
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>
    </div>
  );
}
