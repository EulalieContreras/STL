(self.webpackChunk_jupyter_docprovider_extension=self.webpackChunk_jupyter_docprovider_extension||[]).push([[576],{907:e=>{var t,s,n=e.exports={};function r(){throw new Error("setTimeout has not been defined")}function o(){throw new Error("clearTimeout has not been defined")}function c(e){if(t===setTimeout)return setTimeout(e,0);if((t===r||!t)&&setTimeout)return t=setTimeout,setTimeout(e,0);try{return t(e,0)}catch(s){try{return t.call(null,e,0)}catch(s){return t.call(this,e,0)}}}!function(){try{t="function"==typeof setTimeout?setTimeout:r}catch(e){t=r}try{s="function"==typeof clearTimeout?clearTimeout:o}catch(e){s=o}}();var a,i=[],l=!1,h=-1;function u(){l&&a&&(l=!1,a.length?i=a.concat(i):h=-1,i.length&&d())}function d(){if(!l){var e=c(u);l=!0;for(var t=i.length;t;){for(a=i,i=[];++h<t;)a&&a[h].run();h=-1,t=i.length}a=null,l=!1,function(e){if(s===clearTimeout)return clearTimeout(e);if((s===o||!s)&&clearTimeout)return s=clearTimeout,clearTimeout(e);try{return s(e)}catch(t){try{return s.call(null,e)}catch(t){return s.call(this,e)}}}(e)}}function f(e,t){this.fun=e,this.array=t}function p(){}n.nextTick=function(e){var t=new Array(arguments.length-1);if(arguments.length>1)for(var s=1;s<arguments.length;s++)t[s-1]=arguments[s];i.push(new f(e,t)),1!==i.length||l||c(d)},f.prototype.run=function(){this.fun.apply(null,this.array)},n.title="browser",n.browser=!0,n.env={},n.argv=[],n.version="",n.versions={},n.on=p,n.addListener=p,n.once=p,n.off=p,n.removeListener=p,n.removeAllListeners=p,n.emit=p,n.prependListener=p,n.prependOnceListener=p,n.listeners=function(e){return[]},n.binding=function(e){throw new Error("process.binding is not supported")},n.cwd=function(){return"/"},n.chdir=function(e){throw new Error("process.chdir is not supported")},n.umask=function(){return 0}},576:(e,t,s)=>{"use strict";s.r(t),s.d(t,{WebsocketProvider:()=>Ce,messageAuth:()=>_e,messageAwareness:()=>ve,messageQueryAwareness:()=>ye,messageSync:()=>me});var n=s(206);const r=()=>new Map,o=(e,t,s)=>{let n=e.get(t);return void 0===n&&e.set(t,n=s()),n},c=()=>new Set,a=String.fromCharCode,i=(String.fromCodePoint,a(65535),/^\s*/g),l=/([A-Z])/g,h=(e,t)=>(e=>e.replace(i,""))(e.replace(l,(e=>`${t}${(e=>e.toLowerCase())(e)}`))),u="undefined"!=typeof TextEncoder?new TextEncoder:null,d=u?e=>u.encode(e):e=>{const t=unescape(encodeURIComponent(e)),s=t.length,n=new Uint8Array(s);for(let e=0;e<s;e++)n[e]=t.codePointAt(e);return n};let f="undefined"==typeof TextDecoder?null:new TextDecoder("utf-8",{fatal:!0,ignoreBOM:!0});f&&1===f.decode(new Uint8Array).length&&(f=null);let p=new class{constructor(){this.map=new Map}setItem(e,t){this.map.set(e,t)}getItem(e){return this.map.get(e)}},g=!0;try{"undefined"!=typeof localStorage&&localStorage&&(p=localStorage,g=!1)}catch(e){}const b=p,w=Array.from,m=(Array.isArray,Object.assign,Object.keys),y=e=>m(e).length,v=(e,t)=>Object.prototype.hasOwnProperty.call(e,t),_=(Object.freeze,(e,t)=>{if(null==e||null==t)return((e,t)=>e===t)(e,t);if(e.constructor!==t.constructor)return!1;if(e===t)return!0;switch(e.constructor){case ArrayBuffer:e=new Uint8Array(e),t=new Uint8Array(t);case Uint8Array:if(e.byteLength!==t.byteLength)return!1;for(let s=0;s<e.length;s++)if(e[s]!==t[s])return!1;break;case Set:if(e.size!==t.size)return!1;for(const s of e)if(!t.has(s))return!1;break;case Map:if(e.size!==t.size)return!1;for(const s of e.keys())if(!t.has(s)||!_(e.get(s),t.get(s)))return!1;break;case Object:if(y(e)!==y(t))return!1;for(const s in e)if(!v(e,s)||!_(e[s],t[s]))return!1;break;case Array:if(e.length!==t.length)return!1;for(let s=0;s<e.length;s++)if(!_(e[s],t[s]))return!1;break;default:return!1}return!0});var I=s(907);const S=void 0!==I&&I.release&&/node|io\.js/.test(I.release.name)&&"[object process]"===Object.prototype.toString.call(void 0!==I?I:0),U="undefined"!=typeof window&&"undefined"!=typeof document&&!S;let A;"undefined"!=typeof navigator&&/Mac/.test(navigator.platform);const k=[],C=e=>(()=>{if(void 0===A)if(S){A=r();const e=I.argv;let t=null;for(let s=0;s<e.length;s++){const n=e[s];"-"===n[0]?(null!==t&&A.set(t,""),t=n):null!==t?(A.set(t,n),t=null):k.push(n)}null!==t&&A.set(t,"")}else"object"==typeof location?(A=r(),(location.search||"?").slice(1).split("&").forEach((e=>{if(0!==e.length){const[t,s]=e.split("=");A.set(`--${h(t,"-")}`,s),A.set(`-${h(t,"-")}`,s)}}))):A=r();return A})().has(e),L=e=>{return void 0===(t=S?I.env[e.toUpperCase().replaceAll("-","_")]:b.getItem(e))?null:t;var t},T=e=>C("--"+e)||null!==L(e);var M;T("production"),S&&(M=I.env.FORCE_COLOR,["true","1","2"].includes(M))||!C("--no-colors")&&!T("no-color")&&(!S||I.stdout.isTTY)&&(!S||C("--color")||null!==L("COLORTERM")||(L("TERM")||"").includes("color"));const E=U?e=>{let t="";for(let s=0;s<e.byteLength;s++)t+=a(e[s]);return btoa(t)}:e=>Buffer.from(e.buffer,e.byteOffset,e.byteLength).toString("base64"),O=U?e=>{const t=atob(e),s=(n=t.length,new Uint8Array(n));var n;for(let e=0;e<t.length;e++)s[e]=t.charCodeAt(e);return s}:e=>{const t=Buffer.from(e,"base64");return s=t.buffer,n=t.byteOffset,r=t.byteLength,new Uint8Array(s,n,r);var s,n,r},R=new Map,x="undefined"==typeof BroadcastChannel?class{constructor(e){var t;this.room=e,this.onmessage=null,this._onChange=t=>t.key===e&&null!==this.onmessage&&this.onmessage({data:O(t.newValue||"")}),t=this._onChange,g||addEventListener("storage",t)}postMessage(e){b.setItem(this.room,E(new Uint8Array(e)))}close(){var e;e=this._onChange,g||removeEventListener("storage",e)}}:BroadcastChannel,B=e=>o(R,e,(()=>{const t=c(),s=new x(e);return s.onmessage=e=>t.forEach((t=>t(e.data,"broadcastchannel"))),{bc:s,subs:t}})),D=(e,t,s=null)=>{const n=B(e);n.bc.postMessage(t),n.subs.forEach((e=>e(t,s)))},N=Date.now,j=Math.floor,P=(Math.ceil,Math.abs,Math.imul,Math.round,Math.log10,Math.log2,Math.log,Math.sqrt,(e,t)=>e<t?e:t),H=(Number.isNaN,Math.pow),$=(Math.sign,128),z=127;class W{constructor(){this.cpos=0,this.cbuf=new Uint8Array(100),this.bufs=[]}}const F=()=>new W,V=e=>{let t=e.cpos;for(let s=0;s<e.bufs.length;s++)t+=e.bufs[s].length;return t},G=e=>{const t=new Uint8Array(V(e));let s=0;for(let n=0;n<e.bufs.length;n++){const r=e.bufs[n];t.set(r,s),s+=r.length}return t.set(new Uint8Array(e.cbuf.buffer,0,e.cpos),s),t},J=(e,t)=>{const s=e.cbuf.length;e.cpos===s&&(e.bufs.push(e.cbuf),e.cbuf=new Uint8Array(2*s),e.cpos=0),e.cbuf[e.cpos++]=t},Y=(e,t)=>{for(;t>z;)J(e,$|z&t),t=j(t/128);J(e,z&t)},q=new Uint8Array(3e4),Q=q.length/3,X=u&&u.encodeInto?(e,t)=>{if(t.length<Q){const s=u.encodeInto(t,q).written||0;Y(e,s);for(let t=0;t<s;t++)J(e,q[t])}else Z(e,d(t))}:(e,t)=>{const s=unescape(encodeURIComponent(t)),n=s.length;Y(e,n);for(let t=0;t<n;t++)J(e,s.codePointAt(t))},Z=(e,t)=>{Y(e,t.byteLength),((e,t)=>{const s=e.cbuf.length,n=e.cpos,r=P(s-n,t.length),o=t.length-r;var c,a;e.cbuf.set(t.subarray(0,r),n),e.cpos+=r,o>0&&(e.bufs.push(e.cbuf),e.cbuf=new Uint8Array((c=2*s)>(a=o)?c:a),e.cbuf.set(t.subarray(r)),e.cpos=o)})(e,t)};new DataView(new ArrayBuffer(4));const K=Number.MAX_SAFE_INTEGER,ee=(Number.MIN_SAFE_INTEGER,Number.isInteger,Number.isNaN,Number.parseInt,e=>new Error(e)),te=ee("Unexpected end of array"),se=ee("Integer out of Range");class ne{constructor(e){this.arr=e,this.pos=0}}const re=e=>new ne(e),oe=e=>((e,t)=>{const s=new Uint8Array(e.arr.buffer,e.pos+e.arr.byteOffset,t);return e.pos+=t,s})(e,ae(e)),ce=e=>e.arr[e.pos++],ae=e=>{let t=0,s=1;const n=e.arr.length;for(;e.pos<n;){const n=e.arr[e.pos++];if(t+=(n&z)*s,s*=128,n<$)return t;if(t>K)throw se}throw te},ie=f?e=>f.decode(oe(e)):e=>{let t=ae(e);if(0===t)return"";{let s=String.fromCodePoint(ce(e));if(--t<100)for(;t--;)s+=String.fromCodePoint(ce(e));else for(;t>0;){const n=t<1e4?t:1e4,r=e.arr.subarray(e.pos,e.pos+n);e.pos+=n,s+=String.fromCodePoint.apply(null,r),t-=n}return decodeURIComponent(escape(s))}},le=(e,t)=>{Y(e,0);const s=n.encodeStateVector(t);Z(e,s)},he=(e,t,s)=>{Y(e,1),Z(e,n.encodeStateAsUpdate(t,s))},ue=(e,t,s)=>{try{n.applyUpdate(t,oe(e),s)}catch(e){console.error("Caught error while handling a Yjs update",e)}},de=ue;class fe{constructor(){this._observers=r()}on(e,t){o(this._observers,e,c).add(t)}once(e,t){const s=(...n)=>{this.off(e,s),t(...n)};this.on(e,s)}off(e,t){const s=this._observers.get(e);void 0!==s&&(s.delete(t),0===s.size&&this._observers.delete(e))}emit(e,t){return w((this._observers.get(e)||r()).values()).forEach((e=>e(...t)))}destroy(){this._observers=r()}}class pe extends fe{constructor(e){super(),this.doc=e,this.clientID=e.clientID,this.states=new Map,this.meta=new Map,this._checkInterval=setInterval((()=>{const e=N();null!==this.getLocalState()&&15e3<=e-this.meta.get(this.clientID).lastUpdated&&this.setLocalState(this.getLocalState());const t=[];this.meta.forEach(((s,n)=>{n!==this.clientID&&3e4<=e-s.lastUpdated&&this.states.has(n)&&t.push(n)})),t.length>0&&ge(this,t,"timeout")}),j(3e3)),e.on("destroy",(()=>{this.destroy()})),this.setLocalState({})}destroy(){this.emit("destroy",[this]),this.setLocalState(null),super.destroy(),clearInterval(this._checkInterval)}getLocalState(){return this.states.get(this.clientID)||null}setLocalState(e){const t=this.clientID,s=this.meta.get(t),n=void 0===s?0:s.clock+1,r=this.states.get(t);null===e?this.states.delete(t):this.states.set(t,e),this.meta.set(t,{clock:n,lastUpdated:N()});const o=[],c=[],a=[],i=[];null===e?i.push(t):null==r?null!=e&&o.push(t):(c.push(t),_(r,e)||a.push(t)),(o.length>0||a.length>0||i.length>0)&&this.emit("change",[{added:o,updated:a,removed:i},"local"]),this.emit("update",[{added:o,updated:c,removed:i},"local"])}setLocalStateField(e,t){const s=this.getLocalState();null!==s&&this.setLocalState({...s,[e]:t})}getStates(){return this.states}}const ge=(e,t,s)=>{const n=[];for(let s=0;s<t.length;s++){const r=t[s];if(e.states.has(r)){if(e.states.delete(r),r===e.clientID){const t=e.meta.get(r);e.meta.set(r,{clock:t.clock+1,lastUpdated:N()})}n.push(r)}}n.length>0&&(e.emit("change",[{added:[],updated:[],removed:n},s]),e.emit("update",[{added:[],updated:[],removed:n},s]))},be=(e,t,s=e.states)=>{const n=t.length,r=F();Y(r,n);for(let o=0;o<n;o++){const n=t[o],c=s.get(n)||null,a=e.meta.get(n).clock;Y(r,n),Y(r,a),X(r,JSON.stringify(c))}return G(r)};var we=s(907);const me=0,ye=3,ve=1,_e=2,Ie=[];Ie[me]=(e,t,s,n,r)=>{Y(e,me);const o=((e,t,s,n)=>{const r=ae(e);switch(r){case 0:((e,t,s)=>{he(t,s,oe(e))})(e,t,s);break;case 1:ue(e,s,n);break;case 2:de(e,s,n);break;default:throw new Error("Unknown message type")}return r})(t,e,s.doc,s);n&&1===o&&!s.synced&&(s.synced=!0)},Ie[ye]=(e,t,s,n,r)=>{Y(e,ve),Z(e,be(s.awareness,Array.from(s.awareness.getStates().keys())))},Ie[ve]=(e,t,s,n,r)=>{((e,t,s)=>{const n=re(t),r=N(),o=[],c=[],a=[],i=[],l=ae(n);for(let t=0;t<l;t++){const t=ae(n);let s=ae(n);const l=JSON.parse(ie(n)),h=e.meta.get(t),u=e.states.get(t),d=void 0===h?0:h.clock;(d<s||d===s&&null===l&&e.states.has(t))&&(null===l?t===e.clientID&&null!=e.getLocalState()?s++:e.states.delete(t):e.states.set(t,l),e.meta.set(t,{clock:s,lastUpdated:r}),void 0===h&&null!==l?o.push(t):void 0!==h&&null===l?i.push(t):null!==l&&(_(l,u)||a.push(t),c.push(t)))}(o.length>0||a.length>0||i.length>0)&&e.emit("change",[{added:o,updated:a,removed:i},s]),(o.length>0||c.length>0||i.length>0)&&e.emit("update",[{added:o,updated:c,removed:i},s])})(s.awareness,oe(t),s)},Ie[_e]=(e,t,s,n,r)=>{((e,t,s)=>{0===ae(e)&&s(0,ie(e))})(t,s.doc,((e,t)=>Se(s,t)))};const Se=(e,t)=>console.warn(`Permission denied to access ${e.url}.\n${t}`),Ue=(e,t,s)=>{const n=re(t),r=F(),o=ae(n),c=e.messageHandlers[o];return c?c(r,n,e,s,o):console.error("Unable to compute message"),r},Ae=e=>{if(e.shouldConnect&&null===e.ws){const t=new e._WS(e.url);t.binaryType="arraybuffer",e.ws=t,e.wsconnecting=!0,e.wsconnected=!1,e.synced=!1,t.onmessage=s=>{e.wsLastMessageReceived=N();const n=Ue(e,new Uint8Array(s.data),!0);V(n)>1&&t.send(G(n))},t.onerror=t=>{e.emit("connection-error",[t,e])},t.onclose=t=>{e.emit("connection-close",[t,e]),e.ws=null,e.wsconnecting=!1,e.wsconnected?(e.wsconnected=!1,e.synced=!1,ge(e.awareness,Array.from(e.awareness.getStates().keys()).filter((t=>t!==e.doc.clientID)),e),e.emit("status",[{status:"disconnected"}])):e.wsUnsuccessfulReconnects++,setTimeout(Ae,P(100*H(2,e.wsUnsuccessfulReconnects),e.maxBackoffTime),e)},t.onopen=()=>{e.wsLastMessageReceived=N(),e.wsconnecting=!1,e.wsconnected=!0,e.wsUnsuccessfulReconnects=0,e.emit("status",[{status:"connected"}]);const s=F();if(Y(s,me),le(s,e.doc),t.send(G(s)),null!==e.awareness.getLocalState()){const s=F();Y(s,ve),Z(s,be(e.awareness,[e.doc.clientID])),t.send(G(s))}},e.emit("status",[{status:"connecting"}])}},ke=(e,t)=>{const s=e.ws;e.wsconnected&&s&&s.readyState===s.OPEN&&s.send(t),e.bcconnected&&D(e.bcChannel,t,e)};class Ce extends fe{constructor(e,t,s,{connect:n=!0,awareness:r=new pe(s),params:o={},WebSocketPolyfill:c=WebSocket,resyncInterval:a=-1,maxBackoffTime:i=2500,disableBc:l=!1}={}){for(super();"/"===e[e.length-1];)e=e.slice(0,e.length-1);const h=(e=>((e,t)=>{const s=[];for(const n in e)s.push(t(e[n],n));return s})(e,((e,t)=>`${encodeURIComponent(t)}=${encodeURIComponent(e)}`)).join("&"))(o);this.maxBackoffTime=i,this.bcChannel=e+"/"+t,this.url=e+"/"+t+(0===h.length?"":"?"+h),this.roomname=t,this.doc=s,this._WS=c,this.awareness=r,this.wsconnected=!1,this.wsconnecting=!1,this.bcconnected=!1,this.disableBc=l,this.wsUnsuccessfulReconnects=0,this.messageHandlers=Ie.slice(),this._synced=!1,this.ws=null,this.wsLastMessageReceived=0,this.shouldConnect=n,this._resyncInterval=0,a>0&&(this._resyncInterval=setInterval((()=>{if(this.ws&&this.ws.readyState===WebSocket.OPEN){const e=F();Y(e,me),le(e,s),this.ws.send(G(e))}}),a)),this._bcSubscriber=(e,t)=>{if(t!==this){const t=Ue(this,new Uint8Array(e),!1);V(t)>1&&D(this.bcChannel,G(t),this)}},this._updateHandler=(e,t)=>{if(t!==this){const t=F();Y(t,me),((e,t)=>{Y(e,2),Z(e,t)})(t,e),ke(this,G(t))}},this.doc.on("update",this._updateHandler),this._awarenessUpdateHandler=({added:e,updated:t,removed:s},n)=>{const o=e.concat(t).concat(s),c=F();Y(c,ve),Z(c,be(r,o)),ke(this,G(c))},this._exitHandler=()=>{ge(this.awareness,[s.clientID],"app closed")},S&&void 0!==we&&we.on("exit",this._exitHandler),r.on("update",this._awarenessUpdateHandler),this._checkInterval=setInterval((()=>{this.wsconnected&&3e4<N()-this.wsLastMessageReceived&&this.ws.close()}),3e3),n&&this.connect()}get synced(){return this._synced}set synced(e){this._synced!==e&&(this._synced=e,this.emit("synced",[e]),this.emit("sync",[e]))}destroy(){0!==this._resyncInterval&&clearInterval(this._resyncInterval),clearInterval(this._checkInterval),this.disconnect(),S&&void 0!==we&&we.off("exit",this._exitHandler),this.awareness.off("update",this._awarenessUpdateHandler),this.doc.off("update",this._updateHandler),super.destroy()}connectBc(){if(this.disableBc)return;var e,t;this.bcconnected||(e=this.bcChannel,t=this._bcSubscriber,B(e).subs.add(t),this.bcconnected=!0);const s=F();Y(s,me),le(s,this.doc),D(this.bcChannel,G(s),this);const n=F();Y(n,me),he(n,this.doc),D(this.bcChannel,G(n),this);const r=F();Y(r,ye),D(this.bcChannel,G(r),this);const o=F();Y(o,ve),Z(o,be(this.awareness,[this.doc.clientID])),D(this.bcChannel,G(o),this)}disconnectBc(){const e=F();Y(e,ve),Z(e,be(this.awareness,[this.doc.clientID],new Map)),ke(this,G(e)),this.bcconnected&&(((e,t)=>{const s=B(e);s.subs.delete(t)&&0===s.subs.size&&(s.bc.close(),R.delete(e))})(this.bcChannel,this._bcSubscriber),this.bcconnected=!1)}disconnect(){this.shouldConnect=!1,this.disconnectBc(),null!==this.ws&&this.ws.close()}connect(){this.shouldConnect=!0,this.wsconnected||null!==this.ws||(Ae(this),this.connectBc())}}}}]);