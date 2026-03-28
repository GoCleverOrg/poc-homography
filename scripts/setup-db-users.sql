-- Database user setup for poc-homography
-- These users have been created in Neon PostgreSQL.
-- This script documents their privileges for reference.

-- Database: hom

-- 1. hom_app: Runtime user for the Django application
-- Read-only access to camera_configs, full DML on other tables
CREATE USER hom_app WITH PASSWORD '***';
GRANT CONNECT ON DATABASE hom TO hom_app;
GRANT USAGE ON SCHEMA public TO hom_app;
GRANT SELECT ON camera_configs TO hom_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO hom_app;
REVOKE INSERT, UPDATE, DELETE ON camera_configs FROM hom_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO hom_app;

-- 2. hom_migrate: Migration user for Django schema management
-- Full DDL and DML privileges on all tables
CREATE USER hom_migrate WITH PASSWORD '***';
GRANT CONNECT ON DATABASE hom TO hom_migrate;
GRANT USAGE, CREATE ON SCHEMA public TO hom_migrate;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hom_migrate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hom_migrate;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO hom_migrate;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO hom_migrate;

-- 3. hom_infra: Infra project user for camera config sync
-- Write access only to camera_configs table
CREATE USER hom_infra WITH PASSWORD '***';
GRANT CONNECT ON DATABASE hom TO hom_infra;
GRANT USAGE ON SCHEMA public TO hom_infra;
GRANT SELECT, INSERT, UPDATE, DELETE ON camera_configs TO hom_infra;
