import { useEffect, useState } from 'react';
import { supabase, fetchModels, fetchPredictions, fetchDriftAnalysis, fetchModelMetrics } from '../integrations/supabase/client';
import type { Model, Prediction, DriftAnalysis, ModelMetric } from '../types/supabase';

type TableData = Model[] | Prediction[] | DriftAnalysis[] | ModelMetric[];

interface DataFetcherProps {
  tableName: 'models' | 'predictions' | 'drift_analysis' | 'model_metrics';
  modelId?: string;
}

export const DataFetcher: React.FC<DataFetcherProps> = ({ tableName, modelId }) => {
  const [data, setData] = useState<TableData>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        let result;
        
        switch (tableName) {
          case 'models':
            result = await fetchModels();
            break;
          case 'predictions':
            result = await fetchPredictions(modelId);
            break;
          case 'drift_analysis':
            result = await fetchDriftAnalysis(modelId);
            break;
          case 'model_metrics':
            result = await fetchModelMetrics(modelId);
            break;
        }
        
        setData(result || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [tableName, modelId]);

  // Real-time subscription example
  useEffect(() => {
    const subscription = supabase
      .channel(`${tableName}-changes`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: tableName,
        },
        (payload) => {
          console.log('Change received!', payload);
          // Refresh data when changes occur
          loadData();
        }
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }, [tableName]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">{tableName} Data</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300">
          <thead>
            <tr>
              {data.length > 0 && 
                Object.keys(data[0]).map((key) => (
                  <th key={key} className="px-4 py-2 border-b">{key}</th>
                ))
              }
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                {Object.values(item).map((value, i) => (
                  <td key={i} className="px-4 py-2 border-b">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}; 