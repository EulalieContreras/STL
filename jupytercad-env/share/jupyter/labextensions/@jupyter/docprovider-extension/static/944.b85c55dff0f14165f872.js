"use strict";(self.webpackChunk_jupyter_docprovider_extension=self.webpackChunk_jupyter_docprovider_extension||[]).push([[944],{78:(n,e,o)=>{o.d(e,{A:()=>s});var r=o(758),t=o.n(r),a=o(935),i=o.n(a),l=o(875),p=o(28),c=i()(t());c.i(l.A),c.i(p.A),c.push([n.id,"/* -----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|---------------------------------------------------------------------------- */\n\n.jp-shared-link-body {\n    user-select: none;\n}\n",""]);const s=c},875:(n,e,o)=>{o.d(e,{A:()=>l});var r=o(758),t=o.n(r),a=o(935),i=o.n(a)()(t());i.push([n.id,"/* -----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|---------------------------------------------------------------------------- */\n\n.jp-MenuBar-label {\n  margin-left: 25px;\n}\n\n.jp-MenuBar-anonymousIcon span {\n  width: 24px;\n  text-align: center;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-MenuBar-anonymousIcon,\n.jp-MenuBar-imageIcon {\n  position: absolute;\n  top: 1px;\n  left: 8px;\n  width: 24px;\n  height: 24px;\n  display: flex;\n  align-items: center;\n  vertical-align: middle;\n  border-radius: 100%;\n}\n\n.jp-MenuBar-imageIcon img {\n  width: 24px;\n  border-radius: 100%;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-UserMenu-caretDownIcon {\n  height: 22px;\n  position: relative;\n  top: 15%;\n}\n",""]);const l=i},28:(n,e,o)=>{o.d(e,{A:()=>l});var r=o(758),t=o.n(r),a=o(935),i=o.n(a)()(t());i.push([n.id,"/*\n * Copyright (c) Jupyter Development Team.\n * Distributed under the terms of the Modified BSD License.\n */\n\n/************************************************************\n                      Main Panel\n*************************************************************/\n\n.jp-RTCPanel {\n  min-width: var(--jp-sidebar-min-width) !important;\n  color: var(--jp-ui-font-color1);\n  background: var(--jp-layout-color1);\n  font-size: var(--jp-ui-font-size1);\n}\n\n/************************************************************\n                      User Info Panel\n*************************************************************/\n.jp-UserInfoPanel {\n  display: flex;\n  flex-direction: column;\n  max-height: 140px;\n  padding-top: 3px;\n}\n\n.jp-UserInfo-Container {\n  margin: 20px;\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n}\n\n.jp-UserInfo-Icon {\n  margin: auto;\n  width: 50px;\n  height: 50px;\n  border-radius: 50px;\n  display: inline-flex;\n  align-items: center;\n}\n\n.jp-UserInfo-Icon span {\n  margin: auto;\n  text-align: center;\n  font-size: 25px;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-UserInfo-Info {\n  margin: 20px;\n  display: inline-flex;\n  flex-direction: column;\n}\n\n.jp-UserInfo-Info label {\n  font-weight: bold;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-UserInfo-Info input {\n  text-decoration: none;\n  border-top: none;\n  border-left: none;\n  border-right: none;\n  border-color: var(--jp-ui-font-color1);\n  border-width: 0.5px;\n  background-color: transparent;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n/************************************************************\n                Collaborators Info Panel\n*************************************************************/\n\n.jp-CollaboratorsPanel {\n  overflow-y: auto;\n}\n\n.jp-CollaboratorsList {\n  flex-direction: column;\n  display: flex;\n  z-index: 1000;\n}\n\n.jp-CollaboratorHeader {\n  padding: 10px;\n  display: flex;\n  align-items: center;\n  font-size: var(--jp-ui-font-size0);\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-CollaboratorHeader > span {\n  padding-left: 7px;\n}\n\n.jp-ClickableCollaborator:hover {\n  cursor: pointer;\n  background-color: var(--jp-layout-color2);\n  fill: var(--jp-ui-font-color0);\n  color: var(--jp-ui-font-color0);\n}\n\n.jp-CollaboratorHeaderCollapser {\n  transform: rotate(-90deg);\n  margin: auto 0;\n  height: 16px;\n}\n\n.jp-CollaboratorHeader:not(.jp-ClickableCollaborator) .jp-CollaboratorHeaderCollapser {\n  visibility: hidden;\n}\n\n.jp-CollaboratorHeaderCollapser.jp-mod-expanded {\n  transform: rotate(0deg);\n}\n\n.jp-CollaboratorIcon {\n  border-radius: 100%;\n  padding: 2px;\n  width: 24px;\n  height: 24px;\n  display: flex;\n}\n\n.jp-CollaboratorIcon > span {\n  text-align: center;\n  margin: auto;\n  font-size: 12px;\n  fill: var(--jp-ui-font-color1);\n  color: var(--jp-ui-font-color1);\n}\n\n.jp-CollaboratorFiles {\n  padding-left: 1em;\n  margin-top: 0;\n  box-shadow: 0 2px 2px -2px rgb(0 0 0 / 24%);\n\n}\n",""]);const l=i},935:n=>{n.exports=function(n){var e=[];return e.toString=function(){return this.map((function(e){var o="",r=void 0!==e[5];return e[4]&&(o+="@supports (".concat(e[4],") {")),e[2]&&(o+="@media ".concat(e[2]," {")),r&&(o+="@layer".concat(e[5].length>0?" ".concat(e[5]):""," {")),o+=n(e),r&&(o+="}"),e[2]&&(o+="}"),e[4]&&(o+="}"),o})).join("")},e.i=function(n,o,r,t,a){"string"==typeof n&&(n=[[null,n,void 0]]);var i={};if(r)for(var l=0;l<this.length;l++){var p=this[l][0];null!=p&&(i[p]=!0)}for(var c=0;c<n.length;c++){var s=[].concat(n[c]);r&&i[s[0]]||(void 0!==a&&(void 0===s[5]||(s[1]="@layer".concat(s[5].length>0?" ".concat(s[5]):""," {").concat(s[1],"}")),s[5]=a),o&&(s[2]?(s[1]="@media ".concat(s[2]," {").concat(s[1],"}"),s[2]=o):s[2]=o),t&&(s[4]?(s[1]="@supports (".concat(s[4],") {").concat(s[1],"}"),s[4]=t):s[4]="".concat(t)),e.push(s))}},e}},758:n=>{n.exports=function(n){return n[1]}},591:n=>{var e=[];function o(n){for(var o=-1,r=0;r<e.length;r++)if(e[r].identifier===n){o=r;break}return o}function r(n,r){for(var a={},i=[],l=0;l<n.length;l++){var p=n[l],c=r.base?p[0]+r.base:p[0],s=a[c]||0,u="".concat(c," ").concat(s);a[c]=s+1;var d=o(u),f={css:p[1],media:p[2],sourceMap:p[3],supports:p[4],layer:p[5]};if(-1!==d)e[d].references++,e[d].updater(f);else{var v=t(f,r);r.byIndex=l,e.splice(l,0,{identifier:u,updater:v,references:1})}i.push(u)}return i}function t(n,e){var o=e.domAPI(e);return o.update(n),function(e){if(e){if(e.css===n.css&&e.media===n.media&&e.sourceMap===n.sourceMap&&e.supports===n.supports&&e.layer===n.layer)return;o.update(n=e)}else o.remove()}}n.exports=function(n,t){var a=r(n=n||[],t=t||{});return function(n){n=n||[];for(var i=0;i<a.length;i++){var l=o(a[i]);e[l].references--}for(var p=r(n,t),c=0;c<a.length;c++){var s=o(a[c]);0===e[s].references&&(e[s].updater(),e.splice(s,1))}a=p}}},128:n=>{var e={};n.exports=function(n,o){var r=function(n){if(void 0===e[n]){var o=document.querySelector(n);if(window.HTMLIFrameElement&&o instanceof window.HTMLIFrameElement)try{o=o.contentDocument.head}catch(n){o=null}e[n]=o}return e[n]}(n);if(!r)throw new Error("Couldn't find a style target. This probably means that the value for the 'insert' parameter is invalid.");r.appendChild(o)}},51:n=>{n.exports=function(n){var e=document.createElement("style");return n.setAttributes(e,n.attributes),n.insert(e,n.options),e}},855:(n,e,o)=>{n.exports=function(n){var e=o.nc;e&&n.setAttribute("nonce",e)}},740:n=>{n.exports=function(n){if("undefined"==typeof document)return{update:function(){},remove:function(){}};var e=n.insertStyleElement(n);return{update:function(o){!function(n,e,o){var r="";o.supports&&(r+="@supports (".concat(o.supports,") {")),o.media&&(r+="@media ".concat(o.media," {"));var t=void 0!==o.layer;t&&(r+="@layer".concat(o.layer.length>0?" ".concat(o.layer):""," {")),r+=o.css,t&&(r+="}"),o.media&&(r+="}"),o.supports&&(r+="}");var a=o.sourceMap;a&&"undefined"!=typeof btoa&&(r+="\n/*# sourceMappingURL=data:application/json;base64,".concat(btoa(unescape(encodeURIComponent(JSON.stringify(a))))," */")),e.styleTagTransform(r,n,e.options)}(e,n,o)},remove:function(){!function(n){if(null===n.parentNode)return!1;n.parentNode.removeChild(n)}(e)}}}},656:n=>{n.exports=function(n,e){if(e.styleSheet)e.styleSheet.cssText=n;else{for(;e.firstChild;)e.removeChild(e.firstChild);e.appendChild(document.createTextNode(n))}}},944:(n,e,o)=>{var r=o(591),t=o.n(r),a=o(740),i=o.n(a),l=o(128),p=o.n(l),c=o(855),s=o.n(c),u=o(51),d=o.n(u),f=o(656),v=o.n(f),m=o(78),h={};h.styleTagTransform=v(),h.setAttributes=s(),h.insert=p().bind(null,"head"),h.domAPI=i(),h.insertStyleElement=d(),t()(m.A,h),m.A&&m.A.locals&&m.A.locals}}]);