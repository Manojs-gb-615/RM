// React imported implicitly via JSX transform
import { AlertCircle, CheckCircle } from 'lucide-react';
import type { StatusResponse } from '../api';

export default function StatusBadge({ status }: { status: StatusResponse | null }) {
  if (!status) {
    return (
      <span className="badge bg-gray-700 text-gray-300">
        <span className="w-2 h-2 rounded-full bg-gray-500 animate-pulse" />
        Connecting…
      </span>
    );
  }
  if (status.ready) {
    return (
      <span className="badge bg-green-900 text-green-300">
        <CheckCircle size={12} />
        Ready · {status.num_indexed} poses indexed
      </span>
    );
  }
  return (
    <span className="badge bg-yellow-900 text-yellow-300">
      <AlertCircle size={12} />
      No pose data
    </span>
  );
}
