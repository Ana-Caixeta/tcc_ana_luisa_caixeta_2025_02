# tcc_ana_luisa_caixeta_2025_02

scripts/
│
├── extracao/
│   ├── __init__.py
│   ├── config.py              # Carregamento de configurações e instituições
│   ├── crawler.py             # Funções de requests e scraping
│   ├── processamento.py       # Funções de processamento/transformação dos dados
│   └── main.py                # Orquestrador (executa tudo)
│
└── banco/
    ├── __init__.py
    └── database.py            # Conexão e operações no SQLite
