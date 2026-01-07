-- 1. DROP EXISTING TABLES (To ensure clean state with correct column names)
DROP TABLE IF EXISTS public.files;
DROP TABLE IF EXISTS public.folders;

-- 2. CREATE FOLDERS TABLE (camelCase)
-- checking "companyId" forces mixed case in PostgreSQL
CREATE TABLE public.folders (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "name" TEXT NOT NULL,
    "companyId" TEXT NOT NULL,
    "parentId" UUID REFERENCES public.folders("id") ON DELETE CASCADE,
    "createdBy" UUID REFERENCES auth.users(id),
    "createdAt" TIMESTAMPTZ DEFAULT now(),
    UNIQUE("name", "companyId") -- Prevent duplicate folders in the same company
);

-- 3. CREATE FILES TABLE (camelCase)
CREATE TABLE public.files (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "folderId" UUID REFERENCES public.folders("id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "s3Key" TEXT NOT NULL,
    "s3Url" TEXT NOT NULL,
    "fileType" TEXT NOT NULL,
    "companyId" TEXT NOT NULL,
    "uploadedBy" UUID REFERENCES auth.users(id),
    "createdAt" TIMESTAMPTZ DEFAULT now()
);

-- 4. ENABLE RLS
ALTER TABLE public.folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.files ENABLE ROW LEVEL SECURITY;

-- 5. POLICIES
CREATE POLICY "Users can see their company folders" ON public.folders
    FOR SELECT USING (true); -- In production, filter by "companyId"

CREATE POLICY "Users can create folders" ON public.folders
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can see their company files" ON public.files
    FOR SELECT USING (true);

CREATE POLICY "Users can upload files" ON public.files
    FOR INSERT WITH CHECK (true);

-- 6. FORCE SCHEMA CACHE RELOAD (Required for Supabase API to see changes)
NOTIFY pgrst, 'reload schema';

-- Extra insurance: Commenting on a table also forces cache reload
COMMENT ON TABLE public.folders IS 'Folders for document organization (Schema Reloaded with Quoted Identifiers)';
