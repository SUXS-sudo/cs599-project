CREATE DATABASE IF NOT EXISTS `smart_recipe` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `smart_recipe`;

CREATE TABLE IF NOT EXISTS recipes (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(128) NOT NULL UNIQUE,
  category VARCHAR(64),
  cooking_time_minutes INT,
  difficulty VARCHAR(32),
  calories_per_100g INT,
  protein_g_per_100g DECIMAL(6,1),
  fat_g_per_100g DECIMAL(6,1),
  nutrition_estimated BOOLEAN NOT NULL DEFAULT TRUE,
  steps TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ingredients (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(128) NOT NULL UNIQUE,
  category VARCHAR(64)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS recipe_ingredients (
  recipe_id BIGINT NOT NULL,
  ingredient_id BIGINT NOT NULL,
  amount_text VARCHAR(128),
  PRIMARY KEY (recipe_id, ingredient_id),
  CONSTRAINT fk_recipe_ingredients_recipe
    FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
  CONSTRAINT fk_recipe_ingredients_ingredient
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_indexes (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  index_name VARCHAR(128) NOT NULL UNIQUE,
  index_path VARCHAR(512) NOT NULL,
  metadata_path VARCHAR(512) NOT NULL,
  embedding_backend VARCHAR(512),
  index_type VARCHAR(64),
  hnsw_config JSON,
  chunk_count INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS document_chunks (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  index_name VARCHAR(128) NOT NULL,
  chunk_id VARCHAR(191) NOT NULL,
  source VARCHAR(512),
  source_type VARCHAR(64),
  text MEDIUMTEXT NOT NULL,
  start_char INT,
  end_char INT,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_document_chunks_index_chunk (index_name, chunk_id),
  INDEX idx_document_chunks_index_name (index_name),
  INDEX idx_document_chunks_source (source(191)),
  CONSTRAINT fk_document_chunks_index
    FOREIGN KEY (index_name) REFERENCES document_indexes(index_name) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
