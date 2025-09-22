#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# C:\...\extracao\main.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font
import threading
import asyncio

# Importa dos outros módulos na mesma pasta
from config import INSTITUICOES
from database import DatabaseManager
from scraper import run_for_institution

class ScraperApp(tk.Tk):
    """Classe principal da aplicação com a interface gráfica."""

    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        
        self.title("Integra Scraper")
        self.geometry("800x600")
        self.resizable(False, False)
        self._center_window()
        
        self.create_widgets()
        self.after(100, self.atualizar_tabela_status)

    def _center_window(self):
        """Centraliza a janela na tela."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        """Cria e organiza todos os widgets da interface."""
        frame = ttk.Frame(self, padding=20)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(frame, text="Selecione a instituição:", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        self.combo = ttk.Combobox(frame, values=["TODAS"] + list(INSTITUICOES.keys()), state="readonly", font=("Arial", 10, "bold"), width=30)
        self.combo.pack(pady=(0, 10))
        self.combo.current(0)

        self.btn = tk.Button(frame, text="Iniciar Coleta", bg="#1567BE", fg="white",disabledforeground="white", font=("Arial", 12, "bold"), width=20, height=1, command=self.start_scraping_thread)
        self.btn.pack(pady=(15, 0))

        # --- Barras de Progresso ---
        self._create_progress_bar(frame, "Busca dos professores", "progress_prof", "progress_label_prof_var")
        self._create_progress_bar(frame, "Busca dos detalhes de TCCs", "progress_det", "progress_label_det_var")

        # --- Tabela de Status ---
        ttk.Label(frame, text="Status Geral", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        colunas = ("sigla", "professores", "tccs")
        self.tabela_status = ttk.Treeview(frame, columns=colunas, show="headings", height=8)
        self.tabela_status.heading("sigla", text="Instituição")
        self.tabela_status.heading("professores", text="Total Professores")
        self.tabela_status.heading("tccs", text="Total TCCs")
        self.tabela_status.column("sigla", width=100, anchor="center")
        self.tabela_status.column("professores", width=150, anchor="center")
        self.tabela_status.column("tccs", width=150, anchor="center")
        self.tabela_status.pack(pady=(5, 0), fill="x")

    def _create_progress_bar(self, parent, label_text, progress_attr, label_var_attr):
        """Função auxiliar para criar um conjunto de widgets de progresso."""
        ttk.Label(parent, text=label_text, font=("Arial", 10, "bold")).pack(pady=(10, 2))
        setattr(self, label_var_attr, tk.StringVar(value="0 / 0"))
        ttk.Label(parent, textvariable=getattr(self, label_var_attr)).pack()
        progress_bar = ttk.Progressbar(parent, length=400, mode="determinate")
        progress_bar.pack(pady=(0, 10))
        setattr(self, progress_attr, progress_bar)

    def _update_progress_prof(self, current, total):
        self.progress_prof["maximum"] = total if total != "?" else 100
        self.progress_prof["value"] = current
        self.progress_label_prof_var.set(f"{current} / {total}")

    def _update_progress_det(self, current, total):
        self.progress_det["maximum"] = total
        self.progress_det["value"] = current
        self.progress_label_det_var.set(f"{current} / {total}")
        
    def start_scraping_thread(self):
        """Inicia a coleta em uma nova thread para não bloquear a UI."""
        sigla = self.combo.get()
        if not sigla:
            messagebox.showerror("Erro", "Selecione uma instituição.")
            return

        self.btn.config(state="disabled", text="Coletando...")
        
        # Passa os métodos de atualização da UI como callbacks
        callbacks = {
            'prof_progress': lambda c, t: self.after(0, self._update_progress_prof, c, t),
            'det_progress': lambda c, t: self.after(0, self._update_progress_det, c, t),
        }

        thread = threading.Thread(target=self.run_asyncio_loop, args=(sigla, callbacks), daemon=True)
        thread.start()

    def run_asyncio_loop(self, sigla, callbacks):
        """Executa o loop de eventos asyncio para o scraper."""
        try:
            asyncio.run(self._runner(sigla, callbacks))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro Inesperado", str(e)))
        finally:
            self.after(0, self.scraping_finished, sigla)

    async def _runner(self, sigla, callbacks):
        """Corrotina principal que chama a lógica de scraping."""
        if sigla == "TODAS":
            for s, (url, uf) in INSTITUICOES.items():
                await run_for_institution(s, url, uf, self.db_manager, callbacks)
                self.after(0, self.atualizar_tabela_status)
        else:
            url, uf = INSTITUICOES[sigla]
            await run_for_institution(sigla, url, uf, self.db_manager, callbacks)
        
    def scraping_finished(self, sigla):
        """Chamado quando a coleta termina para reativar o botão e mostrar mensagem."""
        self.btn.config(state="normal", text="Iniciar Coleta")
        self.atualizar_tabela_status()
        messagebox.showinfo("Concluído", f"Coleta finalizada para {sigla}!")

    def atualizar_tabela_status(self):
        """Busca os dados do DB e atualiza a tabela na UI."""
        for i in self.tabela_status.get_children():
            self.tabela_status.delete(i)
        
        dados = self.db_manager.get_status_summary()
        print(dados)
        for sigla, valores in dados["totalizador_uf"].items():
            self.tabela_status.insert(
                "",
                "end",
                values=(sigla, valores["professores"], valores["tccs"])
            )

        # insere linha totalizadora
        default_font = font.nametofont("TkDefaultFont")
        bold_font = font.Font(self, family=default_font.actual("family"),
                        size=default_font.actual("size"),
                        weight="bold")
        self.tabela_status.insert("", "end", values=("TOTAL", dados["total_professores"], dados["total_tccs"]), tags=("bold",))
        self.tabela_status.tag_configure("bold", font=bold_font)

if __name__ == "__main__":
    # Ponto de entrada da aplicação
    db = DatabaseManager()
    db.init_db()  # Garante que o banco e as tabelas existam
    
    app = ScraperApp(db)
    app.mainloop()
