      const STORAGE_KEY = "mon-ia-conversations-v1";
      const conversation = document.querySelector("#conversation");
      const messageList = document.querySelector("#message-list");
      const welcome = document.querySelector("#welcome");
      const historyList = document.querySelector("#history");
      const form = document.querySelector("#composer");
      const promptInput = document.querySelector("#prompt");
      const sendButton = document.querySelector("#send");
      const errorBanner = document.querySelector("#error-banner");
      const documentStatus = document.querySelector("#document-status");
      const pdfList = document.querySelector("#pdf-list");
      const documentUpload = document.querySelector("#document-upload");
      const modelSelect = document.querySelector("#model-select");
      const activeModelLabel = document.querySelector("#active-model");
      const refreshModelsButton = document.querySelector("#refresh-models");
      let chats = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      let activeChatId = null;
      let waiting = false;
      let selectedModel = localStorage.getItem("mon-ia-modele") || "";

      function activeChat() {
        return chats.find((chat) => chat.id === activeChatId);
      }

      function saveChats() {
        const savedChats = chats.map((chat) => ({
          ...chat,
          messages: chat.messages.map(({ role, content }) => ({ role, content })),
        }));
        localStorage.setItem(STORAGE_KEY, JSON.stringify(savedChats));
      }

      function renderHistory() {
        historyList.replaceChildren();
        [...chats].reverse().forEach((chat) => {
          const item = document.createElement("button");
          item.type = "button";
          item.className = `history-item${chat.id === activeChatId ? " active" : ""}`;
          item.textContent = chat.title;
          item.title = chat.title;
          item.addEventListener("click", () => {
            activeChatId = chat.id;
            errorBanner.classList.remove("visible");
            render();
          });
          historyList.append(item);
        });
      }

      function createMessage(message, isError = false) {
        const row = document.createElement("article");
        row.className = `message ${message.role}${isError ? " error" : ""}`;
        if (message.role === "assistant") {
          const avatar = document.createElement("span");
          avatar.className = "message-avatar";
          avatar.setAttribute("aria-hidden", "true");
          avatar.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M12 3.5 14 9l5.5 2-5.5 2-2 5.5L10 13l-5.5-2L10 9l2-5.5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';
          row.append(avatar);
        }
        const body = document.createElement("div");
        body.className = "message-body";
        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = message.content;
        body.append(content);
        if (message.image) {
          const image = document.createElement("img");
          image.className = "message-image";
          image.src = message.image;
          image.alt = "Image générée";
          body.append(image);
        }
        row.append(body);
        return row;
      }

      function render() {
        const chat = activeChat();
        welcome.hidden = Boolean(chat?.messages.length);
        messageList.replaceChildren();
        if (chat) chat.messages.forEach((message) => messageList.append(createMessage(message)));
        renderHistory();
        conversation.scrollTop = conversation.scrollHeight;
      }

      function newChat() {
        activeChatId = null;
        errorBanner.classList.remove("visible");
        render();
        promptInput.focus();
      }

      function resizeInput() {
        promptInput.style.height = "auto";
        promptInput.style.height = `${Math.min(promptInput.scrollHeight, 170)}px`;
      }

      function showError(message) {
        errorBanner.textContent = message;
        errorBanner.classList.add("visible");
      }

      function addTypingIndicator() {
        const row = document.createElement("article");
        row.className = "message assistant";
        row.id = "typing-indicator";
        row.innerHTML = '<span class="message-avatar" aria-hidden="true">✦</span><div class="message-body"><div class="typing" aria-label="Réponse en cours"><i></i><i></i><i></i></div></div>';
        messageList.append(row);
        conversation.scrollTop = conversation.scrollHeight;
      }

      async function lireReponse(response) {
        const body = await response.text();
        try {
          return JSON.parse(body);
        } catch {
          const detail = body.trim().slice(0, 200) || "réponse vide";
          throw new Error(`Réponse non JSON du serveur (HTTP ${response.status}) : ${detail}`);
        }
      }

      function setSelectedModel(model) {
        selectedModel = model;
        modelSelect.value = model;
        activeModelLabel.textContent = model;
        activeModelLabel.title = model;
        localStorage.setItem("mon-ia-modele", selectedModel);
      }

      async function actualiserModeles() {
        refreshModelsButton.disabled = true;
        modelSelect.disabled = true;
        try {
          const response = await fetch("/models");
          const result = await lireReponse(response);
          if (!response.ok) throw new Error(result.detail || "Impossible de lire les modèles disponibles.");
          const models = Array.isArray(result.models)
            ? [...new Set(result.models.filter((model) => typeof model === "string" && model.trim()))]
            : [];
          if (!models.length) throw new Error("Aucun modèle n'est configuré sur le serveur.");
          modelSelect.replaceChildren(...models.map((model) => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            return option;
          }));
          const preferredModel = models.includes(selectedModel) ? selectedModel : models[0];
          setSelectedModel(preferredModel);
          modelSelect.disabled = false;
          errorBanner.classList.remove("visible");
        } finally {
          refreshModelsButton.disabled = false;
        }
      }

      async function actualiserDocuments() {
        const response = await fetch("/documents");
        const result = await lireReponse(response);
        if (!response.ok) throw new Error(result.detail || "Impossible de lire les PDF indexés.");
        const documents = result.documents || [];
        const nombre = documents.length;
        const libelle = `PDF indexés : ${nombre}`;
        documentStatus.textContent = libelle;
        documentStatus.title = documentStatus.textContent;
        pdfList.replaceChildren(...documents.map((pdf) => {
          const item = document.createElement("div");
          item.className = "pdf-item";
          const name = document.createElement("span");
          name.className = "pdf-name";
          name.textContent = pdf.document;
          name.title = `${pdf.document} · ${pdf.pages} page${pdf.pages > 1 ? "s" : ""}`;
          const remove = document.createElement("button");
          remove.className = "pdf-remove";
          remove.type = "button";
          remove.textContent = "Retirer";
          remove.setAttribute("aria-label", `Retirer ${pdf.document} de l'index`);
          remove.addEventListener("click", () => retirerDocument(pdf, remove));
          item.append(name, remove);
          return item;
        }));
      }

      async function retirerDocument(pdf, button) {
        if (!window.confirm(`Retirer « ${pdf.document} » de l'index des PDF ?`)) return;
        button.disabled = true;
        errorBanner.classList.remove("visible");
        try {
          const response = await fetch(`/documents/${encodeURIComponent(pdf.id)}`, {
            method: "DELETE",
          });
          const result = await lireReponse(response);
          if (!response.ok) throw new Error(result.detail || "Impossible de retirer ce PDF.");
          await actualiserDocuments();
        } catch (error) {
          button.disabled = false;
          showError(error instanceof Error ? error.message : "Impossible de retirer ce PDF.");
        }
      }

      documentUpload.addEventListener("change", async () => {
        const fichier = documentUpload.files[0];
        if (!fichier) return;

        documentUpload.disabled = true;
        documentStatus.textContent = `Indexation de ${fichier.name}…`;
        errorBanner.classList.remove("visible");
        try {
          const formData = new FormData();
          formData.append("fichier", fichier);
          const response = await fetch("/documents", { method: "POST", body: formData });
          const result = await lireReponse(response);
          if (!response.ok) throw new Error(result.detail || "Impossible d'indexer ce PDF.");
          await actualiserDocuments();
        } catch (error) {
          showError(error instanceof Error ? error.message : "Impossible d'indexer ce PDF.");
          try {
            await actualiserDocuments();
          } catch (refreshError) {
            console.error(refreshError);
          }
        } finally {
          documentUpload.disabled = false;
          documentUpload.value = "";
        }
      });

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = promptInput.value.trim();
        if (!text || waiting) return;

        if (!activeChat()) {
          const chat = { id: crypto.randomUUID(), title: text.slice(0, 38), messages: [] };
          chats.push(chat);
          activeChatId = chat.id;
        }
        const chat = activeChat();
        chat.messages.push({ role: "user", content: text });
        saveChats();
        render();
        promptInput.value = "";
        resizeInput();
        waiting = true;
        sendButton.disabled = true;
        errorBanner.classList.remove("visible");
        addTypingIndicator();

        try {
          const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: selectedModel,
              messages: chat.messages.map(({ role, content }) => ({ role, content })),
            }),
          });
          const result = await lireReponse(response);
          if (!response.ok) throw new Error(result.detail || "Une erreur est survenue.");
          if (typeof result.reponse !== "string" || !result.reponse.trim()) {
            throw new Error("Le modèle a renvoyé une réponse vide.");
          }
          const sources = Array.isArray(result.sources) && result.sources.length
            ? `\n\nSources : ${result.sources.map((source) => `${source.document}, p. ${source.page}`).join(" · ")}`
            : "";
          chat.messages.push({
            role: "assistant",
            content: `${result.reponse}${sources}`,
            ...(result.image ? { image: result.image } : {}),
          });
          saveChats();
          errorBanner.classList.remove("visible");
        } catch (error) {
          showError(error instanceof Error ? error.message : "Impossible de contacter ton IA. Réessaie.");
        } finally {
          waiting = false;
          sendButton.disabled = false;
          render();
          promptInput.focus();
        }
      });

      if (modelSelect) {
        modelSelect.addEventListener("change", () => setSelectedModel(modelSelect.value));
      }
      refreshModelsButton.addEventListener("click", () => {
        actualiserModeles().catch((error) => {
          showError(error instanceof Error ? error.message : "Impossible de lire les modèles disponibles.");
        });
      });
      document.querySelector("#new-chat").addEventListener("click", newChat);
      document.querySelectorAll(".suggestion").forEach((button) => {
        button.addEventListener("click", () => {
          promptInput.value = button.dataset.prompt;
          resizeInput();
          promptInput.focus();
        });
      });
      promptInput.addEventListener("input", resizeInput);
      promptInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          form.requestSubmit();
        }
      });
      render();
      actualiserModeles().catch((error) => {
        showError(error instanceof Error ? error.message : "Impossible de lire les modèles disponibles.");
      });
      actualiserDocuments().catch((error) => showError(error.message));
