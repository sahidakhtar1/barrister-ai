CREATE SCHEMA IF NOT EXISTS legal;

-- =========================================
-- 1. SOURCE DOCUMENTS
-- =========================================

CREATE TABLE legal.source_documents (
    source_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_url TEXT,
    document_title TEXT,
    publication_date DATE,
    retrieval_date DATE,
    document_hash VARCHAR(128),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT chk_source_type CHECK (
        source_type IN (
            'INDIA_CODE',
            'GAZETTE',
            'GOVERNMENT_WEBSITE',
            'OTHER_OFFICIAL',
            'OTHER'
        )
    )
);


-- =========================================
-- 2. ACTS
-- =========================================

CREATE TABLE legal.acts (
    act_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    canonical_name VARCHAR(500) NOT NULL,
    short_name VARCHAR(255),

    country_code CHAR(2) NOT NULL DEFAULT 'IN',
    jurisdiction VARCHAR(50) NOT NULL DEFAULT 'CENTRAL',

    legislation_type VARCHAR(50) NOT NULL DEFAULT 'ACT',

    act_number VARCHAR(50),
    enactment_year SMALLINT NOT NULL,

    assent_date DATE,
    enactment_date DATE,

    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',

    successor_act_id BIGINT REFERENCES legal.acts(act_id),

    official_source_id BIGINT
        REFERENCES legal.source_documents(source_id),

    description TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_act_identity UNIQUE (
        country_code,
        jurisdiction,
        act_number,
        enactment_year
    ),

    CONSTRAINT chk_act_status CHECK (
        status IN (
            'ACTIVE',
            'REPEALED',
            'PARTIALLY_REPEALED',
            'NOT_YET_IN_FORCE',
            'UNKNOWN'
        )
    ),

    CONSTRAINT chk_jurisdiction CHECK (
        jurisdiction IN ('CENTRAL', 'STATE', 'UNION_TERRITORY')
    )
);


-- =========================================
-- 3. ACT ALIASES
-- =========================================

CREATE TABLE legal.act_aliases (
    alias_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    act_id BIGINT NOT NULL
        REFERENCES legal.acts(act_id) ON DELETE RESTRICT,

    alias VARCHAR(500) NOT NULL,
    normalized_alias VARCHAR(500) NOT NULL,

    alias_type VARCHAR(50) NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_act_alias UNIQUE (
        act_id,
        normalized_alias
    ),

    CONSTRAINT chk_alias_type CHECK (
        alias_type IN (
            'ABBREVIATION',
            'SHORT_NAME',
            'ALTERNATE_NAME',
            'ACT_NUMBER',
            'COMMON_NAME',
            'OTHER'
        )
    )
);


-- =========================================
-- 4. ACT VERSIONS
-- =========================================

CREATE TABLE legal.act_versions (
    act_version_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    act_id BIGINT NOT NULL
        REFERENCES legal.acts(act_id) ON DELETE RESTRICT,

    version_code VARCHAR(100) NOT NULL,
    version_name VARCHAR(255),

    consolidation_date DATE,

    valid_from DATE,
    valid_to DATE,

    is_official BOOLEAN DEFAULT TRUE,

    source_id BIGINT
        REFERENCES legal.source_documents(source_id),

    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_act_version UNIQUE (
        act_id,
        version_code
    ),

    CONSTRAINT uq_act_version_reference UNIQUE (
        act_id,
        act_version_id
    ),

    CONSTRAINT chk_act_version_dates CHECK (
        valid_to IS NULL OR
        valid_from IS NULL OR
        valid_to > valid_from
    )
);


-- =========================================
-- 5. PROVISIONS
-- =========================================

CREATE TABLE legal.provisions (
    provision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    act_id BIGINT NOT NULL
        REFERENCES legal.acts(act_id) ON DELETE RESTRICT,

    parent_provision_id BIGINT,

    provision_type VARCHAR(50) NOT NULL,

    provision_number VARCHAR(100) NOT NULL,

    display_label VARCHAR(255),

    sequence_number INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_provision_act_reference UNIQUE (
        act_id,
        provision_id
    ),

    CONSTRAINT fk_parent_provision
        FOREIGN KEY (act_id, parent_provision_id)
        REFERENCES legal.provisions(act_id, provision_id),

    CONSTRAINT chk_provision_type CHECK (
        provision_type IN (
            'PART',
            'CHAPTER',
            'SECTION',
            'SUBSECTION',
            'CLAUSE',
            'SUBCLAUSE',
            'PROVISO',
            'EXPLANATION',
            'SCHEDULE',
            'ARTICLE',
            'ORDER',
            'RULE',
            'REGULATION',
            'OTHER'
        )
    ),

    CONSTRAINT chk_not_own_parent CHECK (
        parent_provision_id IS NULL OR
        parent_provision_id <> provision_id
    )
);


-- Unique hierarchy position within an Act
CREATE UNIQUE INDEX uq_provision_hierarchy
ON legal.provisions (
    act_id,
    COALESCE(parent_provision_id, 0),
    provision_type,
    provision_number
);


-- =========================================
-- 6. PROVISION VERSIONS
-- =========================================

CREATE TABLE legal.provision_versions (
    provision_version_id BIGINT
        GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    act_id BIGINT NOT NULL,

    provision_id BIGINT NOT NULL,

    act_version_id BIGINT NOT NULL,

    heading TEXT,

    full_text TEXT NOT NULL,

    effective_from DATE,
    effective_to DATE,

    text_hash VARCHAR(128),

    source_id BIGINT
        REFERENCES legal.source_documents(source_id),

    is_inserted BOOLEAN DEFAULT FALSE,

    is_omitted BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_provision_version_act
        FOREIGN KEY (act_id, provision_id)
        REFERENCES legal.provisions(act_id, provision_id),

    CONSTRAINT fk_provision_version_snapshot
        FOREIGN KEY (act_id, act_version_id)
        REFERENCES legal.act_versions(act_id, act_version_id),

    CONSTRAINT uq_provision_version UNIQUE (
        provision_id,
        act_version_id
    ),

    CONSTRAINT chk_provision_version_dates CHECK (
        effective_to IS NULL OR
        effective_from IS NULL OR
        effective_to > effective_from
    )
);


-- =========================================
-- 7. LEGAL EVENTS
-- =========================================

CREATE TABLE legal.legal_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    act_id BIGINT NOT NULL
        REFERENCES legal.acts(act_id),

    provision_id BIGINT,

    related_act_id BIGINT
        REFERENCES legal.acts(act_id),

    event_type VARCHAR(50) NOT NULL,

    event_date DATE,
    effective_date DATE,

    description TEXT,

    source_id BIGINT
        REFERENCES legal.source_documents(source_id),

    additional_metadata JSONB DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_event_provision
        FOREIGN KEY (act_id, provision_id)
        REFERENCES legal.provisions(act_id, provision_id),

    CONSTRAINT chk_legal_event_type CHECK (
        event_type IN (
            'ENACTED',
            'ASSENTED',
            'COMMENCED',
            'AMENDED',
            'INSERTED',
            'SUBSTITUTED',
            'OMITTED',
            'REPEALED',
            'PARTIALLY_REPEALED',
            'SAVED',
            'OTHER'
        )
    )
);


-- =========================================
-- INDEXES FOR CITATION VALIDATION
-- =========================================

CREATE INDEX idx_acts_canonical_name
ON legal.acts USING GIN (
    to_tsvector('english', canonical_name)
);

CREATE INDEX idx_act_aliases_normalized
ON legal.act_aliases(normalized_alias);

CREATE INDEX idx_acts_year
ON legal.acts(enactment_year);

CREATE INDEX idx_act_versions_dates
ON legal.act_versions(act_id, valid_from, valid_to);

CREATE INDEX idx_provisions_lookup
ON legal.provisions(act_id, provision_type, provision_number);

CREATE INDEX idx_provisions_parent
ON legal.provisions(parent_provision_id);

CREATE INDEX idx_provision_versions_lookup
ON legal.provision_versions(provision_id, act_version_id);

CREATE INDEX idx_legal_events_lookup
ON legal.legal_events(act_id, provision_id, effective_date);

CREATE INDEX idx_legal_events_metadata
ON legal.legal_events USING GIN(additional_metadata);