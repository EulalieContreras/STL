"use strict";(self.webpackChunk_jupyter_collaboration_extension=self.webpackChunk_jupyter_collaboration_extension||[]).push([[44],{2044:(e,t,n)=>{n.r(t),n.d(t,{Collaborator:()=>C,CollaboratorsBody:()=>y,CollaboratorsPanel:()=>g,IUserMenu:()=>r,RendererUserMenu:()=>M,UserInfoBody:()=>$,UserInfoPanel:()=>W,UserMenu:()=>N,remoteUserCursors:()=>S,showSharedLinkDialog:()=>B});var o=n(7262);const r=new o.Token("@jupyter/collaboration:IUserMenu");var s=n(5123),a=n(9597),i=n(4602),l=n(5256),c=n(3345),d=n.n(c);const u="jp-CollaboratorsList",h="jp-Collaborator",p="jp-CollaboratorHeader",m="jp-CollaboratorHeaderCollapser",f="jp-ClickableCollaborator",_="jp-CollaboratorIcon",b="jp-CollaboratorFiles",v="jp-CollaboratorFile";class g extends l.Panel{constructor(e,t,n,o){super({}),this._onAwarenessChanged=()=>{const e=this._awareness.getStates(),t=[];e.forEach(((e,n)=>{this._currentUser.isReady&&e.user&&e.user.username!==this._currentUser.identity.username&&t.push(e)})),this._collaboratorsChanged.emit(t)},this._collaboratorsChanged=new i.Signal(this),this._awareness=t,this._currentUser=e,this.addClass("jp-CollaboratorsPanel"),this.addWidget(s.ReactWidget.create(d().createElement(y,{fileopener:n,collaboratorsChanged:this._collaboratorsChanged,docRegistry:o}))),this._awareness.on("change",this._onAwarenessChanged)}}function y(e){const[t,n]=(0,c.useState)([]);return e.collaboratorsChanged.connect(((e,t)=>{n(t)})),d().createElement("div",{className:u},t.map(((t,n)=>d().createElement(C,{collaborator:t,fileopener:e.fileopener,docRegistry:e.docRegistry}))))}function C(e){const[t,n]=(0,c.useState)(!1),{collaborator:o,fileopener:r}=e;let s="";if(o.current){const e=o.current.split(":");s=`${e[1]}:${e[2]}`}const i=o.documents||[],l=i.map((t=>{var n,o;const r=t.split(":"),s=null===(o=null===(n=e.docRegistry)||void 0===n?void 0:n.getFileTypesForPath(r[1]))||void 0===o?void 0:o.filter((e=>void 0!==e.icon)),i=s?s[0].icon:a.fileIcon,l=s?s[0].iconClass:void 0;return{filepath:r[1],filename:r[1].length>40?r[1].slice(0,10).concat("…").concat(r[1].slice(r[1].length-15)):r[1],fileLocation:t,icon:i,iconClass:l}}));return d().createElement("div",{className:h},d().createElement("div",{className:l.length?`${f} ${p}`:p,onClick:i?()=>{l.length&&n(!t)}:void 0},d().createElement(a.LabIcon.resolveReact,{icon:a.caretDownIcon,className:m+(t?" jp-mod-expanded":""),tag:"div"}),d().createElement("div",{className:_,style:{backgroundColor:o.user.color}},d().createElement("span",null,o.user.initials)),d().createElement("span",null,o.user.display_name)),d().createElement("div",{className:`${b} jp-DirListing`,style:t?{}:{display:"none"}},d().createElement("ul",{className:"jp-DirListing-content"},l.map((e=>d().createElement("li",{className:"jp-DirListing-item "+(e.fileLocation===s?`${v} jp-mod-running`:v),key:e.filename,onClick:()=>r(e.fileLocation)},d().createElement(a.LabIcon.resolveReact,{icon:e.icon,iconClass:e.iconClass,tag:"span",className:"jp-DirListing-itemIcon",stylesheet:"listing"}),d().createElement("span",{className:"jp-DirListing-itemText",title:e.filepath},e.filename)))))))}var k=n(195),w=n(5024),j=n(2206);const x=k.Facet.define({combine:e=>e[e.length-1]}),I=w.EditorView.baseTheme({".jp-remote-cursor":{borderLeft:"1px solid black",marginLeft:"-1px"},".jp-remote-cursor.jp-mod-primary":{borderLeftWidth:"2px"},".jp-remote-selection":{opacity:.5},".cm-tooltip":{border:"none"},".cm-tooltip .jp-remote-userInfo":{color:"var(--jp-ui-inverse-font-color0)",padding:"0px 2px"}}),L=k.Annotation.define();class A{constructor(e,t){this.style=e,this.marker=t}draw(){const e=this.marker.draw();for(const[t,n]of Object.entries(this.style))e.style[t]=n;return e}eq(e){return this.marker.eq(e.marker)&&o.JSONExt.deepEqual(this.style,e.style)}update(e,t){for(const[t,n]of Object.entries(this.style))e.style[t]=n;return this.marker.update(e,t.marker)}}const E=(0,w.layer)({above:!0,markers(e){const{awareness:t,ytext:n}=e.state.facet(x),o=n.doc,r=[];return t.getStates().forEach(((s,a)=>{var i,l,c;if(a===t.doc.clientID)return;const d=s.cursors;for(const t of null!=d?d:[]){if(!(null==t?void 0:t.anchor)||!(null==t?void 0:t.head))return;const a=(0,j.createAbsolutePositionFromRelativePosition)(t.anchor,o),d=(0,j.createAbsolutePositionFromRelativePosition)(t.head,o);if((null==a?void 0:a.type)!==n||(null==d?void 0:d.type)!==n)return;const u=null===(i=t.primary)||void 0===i||i?"jp-remote-cursor jp-mod-primary":"jp-remote-cursor",h=k.EditorSelection.cursor(d.index,d.index>a.index?-1:1);for(const t of w.RectangleMarker.forRange(e,u,h))r.push(new A({borderLeftColor:null!==(c=null===(l=s.user)||void 0===l?void 0:l.color)&&void 0!==c?c:"black"},t))}})),r},update:(e,t)=>!!e.transactions.find((e=>e.annotation(L))),class:"jp-remote-cursors"}),R=(0,w.hoverTooltip)(((e,t)=>{var n;const{awareness:o,ytext:r}=e.state.facet(x),s=r.doc;for(const[e,a]of o.getStates())if(e!==o.doc.clientID)for(const e of null!==(n=a.cursors)&&void 0!==n?n:[]){if(!(null==e?void 0:e.head))continue;const n=(0,j.createAbsolutePositionFromRelativePosition)(e.head,s);if((null==n?void 0:n.type)===r&&n.index-3<=t&&t<=n.index+3)return{pos:n.index,above:!0,create:()=>{var e,t,n,o;const r=document.createElement("div");return r.classList.add("jp-remote-userInfo"),r.style.backgroundColor=null!==(t=null===(e=a.user)||void 0===e?void 0:e.color)&&void 0!==t?t:"darkgrey",r.textContent=null!==(o=null===(n=a.user)||void 0===n?void 0:n.display_name)&&void 0!==o?o:"Anonymous",{dom:r}}}}return null}),{hideOn:(e,t)=>!!e.annotation(L),hoverTime:0}),P=(0,w.layer)({above:!1,markers(e){const{awareness:t,ytext:n}=e.state.facet(x),o=n.doc,r=[];return t.getStates().forEach(((s,a)=>{var i,l,c;if(a===t.doc.clientID)return;const d=s.cursors;for(const t of null!=d?d:[]){if(null===(i=t.empty)||void 0===i||i||!(null==t?void 0:t.anchor)||!(null==t?void 0:t.head))return;const a=(0,j.createAbsolutePositionFromRelativePosition)(t.anchor,o),d=(0,j.createAbsolutePositionFromRelativePosition)(t.head,o);if((null==a?void 0:a.type)!==n||(null==d?void 0:d.type)!==n)return;const u="jp-remote-selection";for(const t of w.RectangleMarker.forRange(e,u,k.EditorSelection.range(a.index,d.index)))r.push(new A({backgroundColor:null!==(c=null===(l=s.user)||void 0===l?void 0:l.color)&&void 0!==c?c:"black"},t))}})),r},update:(e,t)=>!!e.transactions.find((e=>e.annotation(L))),class:"jp-remote-selections"}),T=w.ViewPlugin.fromClass(class{constructor(e){this.editorAwareness=e.state.facet(x),this._listener=({added:t,updated:n,removed:o})=>{t.concat(n).concat(o).findIndex((e=>e!==this.editorAwareness.awareness.doc.clientID))>=0&&e.dispatch({annotations:[L.of([])]})},this.editorAwareness.awareness.on("change",this._listener)}destroy(){this.editorAwareness.awareness.off("change",this._listener)}update(e){var t;if(!e.docChanged&&!e.selectionSet)return;const{awareness:n,ytext:r}=this.editorAwareness,s=n.getLocalState();if(s){const a=e.view.hasFocus&&e.view.dom.ownerDocument.hasFocus(),i=e.state.selection,l=new Array;if(a&&i){for(const e of i.ranges){const t=e===i.main,n=(0,j.createRelativePositionFromTypeIndex)(r,e.anchor),o=(0,j.createRelativePositionFromTypeIndex)(r,e.head);l.push({anchor:n,head:o,primary:t,empty:e.empty})}if(!s.cursors||l.length>0){const e=null===(t=s.cursors)||void 0===t?void 0:t.map((e=>({...e,anchor:(null==e?void 0:e.anchor)?(0,j.createRelativePositionFromJSON)(e.anchor):null,head:(null==e?void 0:e.head)?(0,j.createRelativePositionFromJSON)(e.head):null})));o.JSONExt.deepEqual(l,e)||n.setLocalStateField("cursors",l)}}}}},{provide:()=>[I,E,P,R,(0,w.tooltips)({position:"absolute",parent:document.body})]});function S(e){return[x.of(e),T]}var U=n(4873);class M extends l.MenuBar.Renderer{constructor(e){super(),this._user=e}renderItem(e){const t=this.createItemClass(e),n=this.createItemDataset(e),o=this.createItemARIA(e);return U.h.li({className:t,dataset:n,tabindex:"0",onfocus:e.onfocus,...o},this._createUserIcon(),this.renderLabel(e),this.renderIcon(e))}renderLabel(e){const t=this.formatLabel(e);return U.h.div({className:"lm-MenuBar-itemLabel jp-MenuBar-label"},t)}_createUserIcon(){return this._user.isReady&&this._user.identity.avatar_url?U.h.div({className:"lm-MenuBar-itemIcon jp-MenuBar-imageIcon"},U.h.img({src:this._user.identity.avatar_url})):this._user.isReady?U.h.div({className:"lm-MenuBar-itemIcon jp-MenuBar-anonymousIcon",style:{backgroundColor:this._user.identity.color}},U.h.span({},this._user.identity.initials)):U.h.div({className:"lm-MenuBar-itemIcon jp-MenuBar-anonymousIcon"},a.userIcon)}}class N extends l.Menu{constructor(e){super(e)}}var D=n(3672),F=n(1243);async function B({translator:e}){const t=(null!=e?e:F.nullTranslator).load("collaboration"),n=D.PageConfig.getToken(),o=new URL(D.URLExt.normalize(D.PageConfig.getUrl({workspace:D.PageConfig.defaultWorkspace})));return(0,s.showDialog)({title:t.__("Share Jupyter Server Link"),body:new H(o.toString(),n,""!==D.PageConfig.getOption("hubUser"),t),buttons:[s.Dialog.cancelButton(),s.Dialog.okButton({label:t.__("Copy Link"),caption:t.__("Copy the link to the Jupyter Server")})]})}class H extends l.Widget{constructor(e,t,n,o){super(),this._url=e,this._token=t,this._behindHub=n,this._trans=o,this._tokenCheckbox=null,this.onTokenChange=e=>{const t=e.target;this.updateContent(null==t?void 0:t.checked)},this._warning=document.createElement("div"),this.populateBody(this.node),this.addClass("jp-shared-link-body")}getValue(){var e;if(!0===(null===(e=this._tokenCheckbox)||void 0===e?void 0:e.checked)){const e=new URL(this._url);return e.searchParams.set("token",this._token),e.toString()}return this._url}onAfterAttach(e){var t;super.onAfterAttach(e),null===(t=this._tokenCheckbox)||void 0===t||t.addEventListener("change",this.onTokenChange)}onBeforeDetach(e){var t;null===(t=this._tokenCheckbox)||void 0===t||t.removeEventListener("change",this.onTokenChange),super.onBeforeDetach(e)}updateContent(e){this._warning.innerHTML="";const t=this.node.querySelector("input[readonly]");if(e){if(t){const e=new URL(this._url);e.searchParams.set("token",this._token.slice(0,5)),t.value=e.toString()+"…"}this._warning.appendChild(document.createElement("h3")).textContent=this._trans.__("Security warning!"),this._warning.insertAdjacentText("beforeend",this._trans.__("Anyone with this link has full access to your notebook server, including all your files!")),this._warning.insertAdjacentHTML("beforeend","<br>"),this._warning.insertAdjacentText("beforeend",this._trans.__("Please be careful who you share it with.")),this._warning.insertAdjacentHTML("beforeend","<br>"),this._behindHub?(this._warning.insertAdjacentText("beforeend",this._trans.__("They will be able to access this server AS YOU.")),this._warning.insertAdjacentHTML("beforeend","<br>"),this._warning.insertAdjacentText("beforeend",this._trans.__("To revoke access, go to File -> Hub Control Panel, and restart your server."))):this._warning.insertAdjacentText("beforeend",this._trans.__("Currently, there is no way to revoke access other than shutting down your server."))}else t&&(t.value=this._url),this._behindHub?this._warning.insertAdjacentText("beforeend",this._trans.__("Only users with `access:servers` permissions for this server will be able to use this link.")):this._warning.insertAdjacentText("beforeend",this._trans.__("Only authenticated users will be able to use this link."))}populateBody(e){if(e.insertAdjacentHTML("afterbegin",`<input readonly value="${this._url}">`),this._token){const t=e.appendChild(document.createElement("label"));t.insertAdjacentHTML("beforeend",'<input type="checkbox">'),this._tokenCheckbox=t.firstChild,t.insertAdjacentText("beforeend",this._trans.__("Include token in URL")),e.insertAdjacentElement("beforeend",this._warning),this.updateContent(!1)}}}const O=e=>{const{user:t}=e;return c.createElement("div",{className:"jp-UserInfo-Container"},c.createElement("div",{title:t.display_name,className:"jp-UserInfo-Icon",style:{backgroundColor:t.color}},c.createElement("span",null,t.initials)),c.createElement("h3",null,t.display_name))};class W extends l.Panel{constructor(e){super({}),this.addClass("jp-UserInfoPanel"),this._profile=e,this._body=null,this._profile.isReady?(this._body=new $(this._profile.identity),this.addWidget(this._body),this.update()):this._profile.ready.then((()=>{this._body=new $(this._profile.identity),this.addWidget(this._body),this.update()})).catch((e=>console.error(e)))}}class $ extends s.ReactWidget{constructor(e){super(),this._user=e}get user(){return this._user}set user(e){this._user=e,this.update()}render(){return c.createElement(O,{user:this._user})}}}}]);