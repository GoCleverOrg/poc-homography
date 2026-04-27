import { useTenant } from '../contexts/TenantContext';

export default function TenantSelector() {
  const { tenants, selectedTenantId, setSelectedTenantId, loading, error } =
    useTenant();

  if (loading) return <span>Loading tenants...</span>;
  if (error) return <span>Error: {error}</span>;
  if (!tenants || tenants.length === 0) return <span>No tenants available</span>;

  return (
    <select
      value={selectedTenantId ?? ''}
      onChange={(e) => setSelectedTenantId(e.target.value)}
    >
      {tenants.map((t) => (
        <option key={t.id} value={t.id}>
          {t.name}
          {t.description ? ` - ${t.description}` : ''}
        </option>
      ))}
    </select>
  );
}
