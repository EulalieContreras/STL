(self.webpackChunk_jupyter_collaboration_extension=self.webpackChunk_jupyter_collaboration_extension||[]).push([[952],{9907:e=>{var t,n,s=e.exports={};function o(){throw new Error("setTimeout has not been defined")}function c(){throw new Error("clearTimeout has not been defined")}function r(e){if(t===setTimeout)return setTimeout(e,0);if((t===o||!t)&&setTimeout)return t=setTimeout,setTimeout(e,0);try{return t(e,0)}catch(n){try{return t.call(null,e,0)}catch(n){return t.call(this,e,0)}}}!function(){try{t="function"==typeof setTimeout?setTimeout:o}catch(e){t=o}try{n="function"==typeof clearTimeout?clearTimeout:c}catch(e){n=c}}();var a,i=[],l=!1,h=-1;function d(){l&&a&&(l=!1,a.length?i=a.concat(i):h=-1,i.length&&u())}function u(){if(!l){var e=r(d);l=!0;for(var t=i.length;t;){for(a=i,i=[];++h<t;)a&&a[h].run();h=-1,t=i.length}a=null,l=!1,function(e){if(n===clearTimeout)return clearTimeout(e);if((n===c||!n)&&clearTimeout)return n=clearTimeout,clearTimeout(e);try{return n(e)}catch(t){try{return n.call(null,e)}catch(t){return n.call(this,e)}}}(e)}}function f(e,t){this.fun=e,this.array=t}function w(){}s.nextTick=function(e){var t=new Array(arguments.length-1);if(arguments.length>1)for(var n=1;n<arguments.length;n++)t[n-1]=arguments[n];i.push(new f(e,t)),1!==i.length||l||r(u)},f.prototype.run=function(){this.fun.apply(null,this.array)},s.title="browser",s.browser=!0,s.env={},s.argv=[],s.version="",s.versions={},s.on=w,s.addListener=w,s.once=w,s.off=w,s.removeListener=w,s.removeAllListeners=w,s.emit=w,s.prependListener=w,s.prependOnceListener=w,s.listeners=function(e){return[]},s.binding=function(e){throw new Error("process.binding is not supported")},s.cwd=function(){return"/"},s.chdir=function(e){throw new Error("process.chdir is not supported")},s.umask=function(){return 0}},2952:(e,t,n)=>{"use strict";n.r(t),n.d(t,{WebsocketProvider:()=>Q,messageAuth:()=>$,messageAwareness:()=>G,messageQueryAwareness:()=>O,messageSync:()=>H});var s=n(2206),o=n(6505),c=n(9279),r=n(1554);let a=new class{constructor(){this.map=new Map}setItem(e,t){this.map.set(e,t)}getItem(e){return this.map.get(e)}},i=!0;try{"undefined"!=typeof localStorage&&localStorage&&(a=localStorage,i=!1)}catch(e){}const l=a;var h=n(6527),d=n(9907);const u=void 0!==d&&d.release&&/node|io\.js/.test(d.release.name)&&"[object process]"===Object.prototype.toString.call(void 0!==d?d:0),f="undefined"!=typeof window&&"undefined"!=typeof document&&!u;let w;"undefined"!=typeof navigator&&/Mac/.test(navigator.platform);const p=[],b=e=>(()=>{if(void 0===w)if(u){w=o.vt();const e=d.argv;let t=null;for(let n=0;n<e.length;n++){const s=e[n];"-"===s[0]?(null!==t&&w.set(t,""),t=s):null!==t?(w.set(t,s),t=null):p.push(s)}null!==t&&w.set(t,"")}else"object"==typeof location?(w=o.vt(),(location.search||"?").slice(1).split("&").forEach((e=>{if(0!==e.length){const[t,n]=e.split("=");w.set(`--${r.jN(t,"-")}`,n),w.set(`-${r.jN(t,"-")}`,n)}}))):w=o.vt();return w})().has(e),g=e=>{return void 0===(t=u?d.env[e.toUpperCase().replaceAll("-","_")]:l.getItem(e))?null:t;var t},m=e=>b("--"+e)||null!==g(e),y=(m("production"),u&&h.EK(d.env.FORCE_COLOR,["true","1","2"])||!b("--no-colors")&&!m("no-color")&&(!u||d.stdout.isTTY)&&(!u||b("--color")||null!==g("COLORTERM")||(g("TERM")||"").includes("color")),e=>new Uint8Array(e)),v=f?e=>{let t="";for(let n=0;n<e.byteLength;n++)t+=r.QV(e[n]);return btoa(t)}:e=>Buffer.from(e.buffer,e.byteOffset,e.byteLength).toString("base64"),_=f?e=>{const t=atob(e),n=y(t.length);for(let e=0;e<t.length;e++)n[e]=t.charCodeAt(e);return n}:e=>{const t=Buffer.from(e,"base64");return n=t.buffer,s=t.byteOffset,o=t.byteLength,new Uint8Array(n,s,o);var n,s,o},C=new Map,T="undefined"==typeof BroadcastChannel?class{constructor(e){var t;this.room=e,this.onmessage=null,this._onChange=t=>t.key===e&&null!==this.onmessage&&this.onmessage({data:_(t.newValue||"")}),t=this._onChange,i||addEventListener("storage",t)}postMessage(e){l.setItem(this.room,v(new Uint8Array(e)))}close(){var e;e=this._onChange,i||removeEventListener("storage",e)}}:BroadcastChannel,x=e=>o._4(C,e,(()=>{const t=c.vt(),n=new T(e);return n.onmessage=e=>t.forEach((t=>t(e.data,"broadcastchannel"))),{bc:n,subs:t}})),k=(e,t,n=null)=>{const s=x(e);s.bc.postMessage(t),s.subs.forEach((e=>e(t,n)))};var I=n(482),S=n(6214),U=n(3721);const L=(e,t)=>{S.zd(e,0);const n=s.encodeStateVector(t);S.Gu(e,n)},z=(e,t,n)=>{S.zd(e,1),S.Gu(e,s.encodeStateAsUpdate(t,n))},B=(e,t,n)=>{try{s.applyUpdate(t,U.bo(e),n)}catch(e){console.error("Caught error while handling a Yjs update",e)}},E=B;var A=n(7784),R=n(1370),F=n(801),j=n(6132),M=n(9907);const H=0,O=3,G=1,$=2,W=[];W[H]=(e,t,n,s,o)=>{S.zd(e,H);const c=((e,t,n,s)=>{const o=U.cw(e);switch(o){case 0:((e,t,n)=>{z(t,n,U.bo(e))})(e,t,n);break;case 1:B(e,n,s);break;case 2:E(e,n,s);break;default:throw new Error("Unknown message type")}return o})(t,e,n.doc,n);s&&1===c&&!n.synced&&(n.synced=!0)},W[O]=(e,t,n,s,o)=>{S.zd(e,G),S.Gu(e,A.X3(n.awareness,Array.from(n.awareness.getStates().keys())))},W[G]=(e,t,n,s,o)=>{A.tQ(n.awareness,U.bo(t),n)},W[$]=(e,t,n,s,o)=>{((e,t,n)=>{0===U.cw(e)&&n(0,U.t3(e))})(t,n.doc,((e,t)=>D(n,t)))};const D=(e,t)=>console.warn(`Permission denied to access ${e.url}.\n${t}`),P=(e,t,n)=>{const s=U.$C(t),o=S.xv(),c=U.cw(s),r=e.messageHandlers[c];return r?r(o,s,e,n,c):console.error("Unable to compute message"),o},X=e=>{if(e.shouldConnect&&null===e.ws){const t=new e._WS(e.url);t.binaryType="arraybuffer",e.ws=t,e.wsconnecting=!0,e.wsconnected=!1,e.synced=!1,t.onmessage=n=>{e.wsLastMessageReceived=I._g();const s=P(e,new Uint8Array(n.data),!0);S.Bw(s)>1&&t.send(S.Fo(s))},t.onerror=t=>{e.emit("connection-error",[t,e])},t.onclose=t=>{e.emit("connection-close",[t,e]),e.ws=null,e.wsconnecting=!1,e.wsconnected?(e.wsconnected=!1,e.synced=!1,A._g(e.awareness,Array.from(e.awareness.getStates().keys()).filter((t=>t!==e.doc.clientID)),e),e.emit("status",[{status:"disconnected"}])):e.wsUnsuccessfulReconnects++,setTimeout(X,F.jk(100*F.n7(2,e.wsUnsuccessfulReconnects),e.maxBackoffTime),e)},t.onopen=()=>{e.wsLastMessageReceived=I._g(),e.wsconnecting=!1,e.wsconnected=!0,e.wsUnsuccessfulReconnects=0,e.emit("status",[{status:"connected"}]);const n=S.xv();if(S.zd(n,H),L(n,e.doc),t.send(S.Fo(n)),null!==e.awareness.getLocalState()){const n=S.xv();S.zd(n,G),S.Gu(n,A.X3(e.awareness,[e.doc.clientID])),t.send(S.Fo(n))}},e.emit("status",[{status:"connecting"}])}},N=(e,t)=>{const n=e.ws;e.wsconnected&&n&&n.readyState===n.OPEN&&n.send(t),e.bcconnected&&k(e.bcChannel,t,e)};class Q extends R.c{constructor(e,t,n,{connect:s=!0,awareness:o=new A.ww(n),params:c={},WebSocketPolyfill:r=WebSocket,resyncInterval:a=-1,maxBackoffTime:i=2500,disableBc:l=!1}={}){for(super();"/"===e[e.length-1];)e=e.slice(0,e.length-1);const h=(e=>j.Tj(e,((e,t)=>`${encodeURIComponent(t)}=${encodeURIComponent(e)}`)).join("&"))(c);this.maxBackoffTime=i,this.bcChannel=e+"/"+t,this.url=e+"/"+t+(0===h.length?"":"?"+h),this.roomname=t,this.doc=n,this._WS=r,this.awareness=o,this.wsconnected=!1,this.wsconnecting=!1,this.bcconnected=!1,this.disableBc=l,this.wsUnsuccessfulReconnects=0,this.messageHandlers=W.slice(),this._synced=!1,this.ws=null,this.wsLastMessageReceived=0,this.shouldConnect=s,this._resyncInterval=0,a>0&&(this._resyncInterval=setInterval((()=>{if(this.ws&&this.ws.readyState===WebSocket.OPEN){const e=S.xv();S.zd(e,H),L(e,n),this.ws.send(S.Fo(e))}}),a)),this._bcSubscriber=(e,t)=>{if(t!==this){const t=P(this,new Uint8Array(e),!1);S.Bw(t)>1&&k(this.bcChannel,S.Fo(t),this)}},this._updateHandler=(e,t)=>{if(t!==this){const t=S.xv();S.zd(t,H),((e,t)=>{S.zd(e,2),S.Gu(e,t)})(t,e),N(this,S.Fo(t))}},this.doc.on("update",this._updateHandler),this._awarenessUpdateHandler=({added:e,updated:t,removed:n},s)=>{const c=e.concat(t).concat(n),r=S.xv();S.zd(r,G),S.Gu(r,A.X3(o,c)),N(this,S.Fo(r))},this._exitHandler=()=>{A._g(this.awareness,[n.clientID],"app closed")},u&&void 0!==M&&M.on("exit",this._exitHandler),o.on("update",this._awarenessUpdateHandler),this._checkInterval=setInterval((()=>{this.wsconnected&&3e4<I._g()-this.wsLastMessageReceived&&this.ws.close()}),3e3),s&&this.connect()}get synced(){return this._synced}set synced(e){this._synced!==e&&(this._synced=e,this.emit("synced",[e]),this.emit("sync",[e]))}destroy(){0!==this._resyncInterval&&clearInterval(this._resyncInterval),clearInterval(this._checkInterval),this.disconnect(),u&&void 0!==M&&M.off("exit",this._exitHandler),this.awareness.off("update",this._awarenessUpdateHandler),this.doc.off("update",this._updateHandler),super.destroy()}connectBc(){if(this.disableBc)return;var e,t;this.bcconnected||(e=this.bcChannel,t=this._bcSubscriber,x(e).subs.add(t),this.bcconnected=!0);const n=S.xv();S.zd(n,H),L(n,this.doc),k(this.bcChannel,S.Fo(n),this);const s=S.xv();S.zd(s,H),z(s,this.doc),k(this.bcChannel,S.Fo(s),this);const o=S.xv();S.zd(o,O),k(this.bcChannel,S.Fo(o),this);const c=S.xv();S.zd(c,G),S.Gu(c,A.X3(this.awareness,[this.doc.clientID])),k(this.bcChannel,S.Fo(c),this)}disconnectBc(){const e=S.xv();S.zd(e,G),S.Gu(e,A.X3(this.awareness,[this.doc.clientID],new Map)),N(this,S.Fo(e)),this.bcconnected&&(((e,t)=>{const n=x(e);n.subs.delete(t)&&0===n.subs.size&&(n.bc.close(),C.delete(e))})(this.bcChannel,this._bcSubscriber),this.bcconnected=!1)}disconnect(){this.shouldConnect=!1,this.disconnectBc(),null!==this.ws&&this.ws.close()}connect(){this.shouldConnect=!0,this.wsconnected||null!==this.ws||(X(this),this.connectBc())}}}}]);