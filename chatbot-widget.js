/**
 * CRSIJ Chatbot Widget
 * Upload this file to your website (e.g. as /chatbot-widget.js)
 * Also upload chatbot-theme.css and crsi-logo-trimmed.png to the same folder.
 * Then add this single line before your closing </body> tag on any page you want the chatbot to appear:
 *
 *   <script src="/chatbot-widget.js"></script>
 *
 * That's it — no other changes needed to your existing site.
 */

(function () {
  // 1. Load the chat widget's default stylesheet (from n8n's CDN)
  const baseStyle = document.createElement('link');
  baseStyle.rel = 'stylesheet';
  baseStyle.href = 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/style.css';
  document.head.appendChild(baseStyle);

  // 2. Load YOUR custom theme CSS file - this replaces all the inline
  // <style> JS code from before. Update the href below if you host
  // chatbot-theme.css somewhere other than your site's root.
  const themeStyle = document.createElement('link');
  themeStyle.rel = 'stylesheet';
  themeStyle.href = '/chatbot-theme.css';
  document.head.appendChild(themeStyle);

  // 3. Load the chat widget script as a module, then initialize it
  const script = document.createElement('script');
  script.type = 'module';
  script.textContent = `
    import { createChat } from 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/chat.bundle.es.js';
    createChat({
      webhookUrl: 'https://chatbot-seven-ebon-56.vercel.app/webhook/chat',
      initialMessages: [
        "Hi there! \u{1F44B} I'm the CRSI Journal assistant. Ask me about submitting a paper, tracking your paper, or publication charges!"
      ],
      i18n: {
        en: {
          title: 'CRSIJ Sathi',
          subtitle: '',
          footer: '',
          getStarted: 'New Conversation',
          inputPlaceholder: 'Type your question...',
        },
      },
    });
  `;
  document.body.appendChild(script);

  // 4. Add the CRSI logo + title/subtitle to the widget header once it appears.
  const observer = new MutationObserver(() => {
    const header = document.querySelector('.chat-header');
    if (header && !header.querySelector('.crsi-header-wrap')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'crsi-header-wrap';
      const logo = document.createElement('img');
      logo.src = '/crsi-logo-square.png';
      logo.alt = 'CRSI Journal';
      logo.className = 'crsi-logo-img';
      const textWrap = document.createElement('div');
      textWrap.className = 'crsi-text-wrap';
      const title = document.createElement('div');
      title.className = 'crsi-main-title';
      title.textContent = 'CRSI Sathi';
      const subtitle = document.createElement('div');
      subtitle.className = 'crsi-subtitle';
      subtitle.textContent = 'AI Journal Assistant';
      textWrap.appendChild(title);
      textWrap.appendChild(subtitle);
      wrapper.appendChild(logo);
      wrapper.appendChild(textWrap);
      // DON'T replace the entire header
      header.prepend(wrapper);
      observer.disconnect();
    }
  });
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
})();