-- Add fields and messages columns for session restoration
ALTER TABLE form_states
ADD COLUMN IF NOT EXISTS fields JSONB DEFAULT '[]'::jsonb,
ADD COLUMN IF NOT EXISTS messages JSONB DEFAULT '[]'::jsonb;

-- Add index for faster queries
CREATE INDEX IF NOT EXISTS idx_form_states_session_id ON form_states(session_id);
