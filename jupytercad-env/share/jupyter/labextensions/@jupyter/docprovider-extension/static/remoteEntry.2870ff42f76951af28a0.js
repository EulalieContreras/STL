var _JUPYTERLAB;(()=>{"use strict";var e,r,t,o,a,n,i,l,u,s,d,f,p,c,v,h,b,y,m,g,j,w,P,S={496:(e,r,t)=>{var o={"./index":()=>Promise.all([t.e(260),t.e(401),t.e(703)]).then((()=>()=>t(703))),"./extension":()=>Promise.all([t.e(260),t.e(401),t.e(703)]).then((()=>()=>t(703))),"./style":()=>t.e(944).then((()=>()=>t(944)))},a=(e,r)=>(t.R=r,r=t.o(o,e)?o[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),n=(e,r)=>{if(t.S){var o="default",a=t.S[o];if(a&&a!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[o]=e,t.I(o,r)}};t.d(r,{get:()=>a,init:()=>n})}},k={};function x(e){var r=k[e];if(void 0!==r)return r.exports;var t=k[e]={id:e,exports:{}};return S[e](t,t.exports,x),t.exports}x.m=S,x.c=k,x.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return x.d(r,{a:r}),r},x.d=(e,r)=>{for(var t in r)x.o(r,t)&&!x.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},x.f={},x.e=e=>Promise.all(Object.keys(x.f).reduce(((r,t)=>(x.f[t](e,r),r)),[])),x.u=e=>e+"."+{240:"dd1dbd559d6c95c85b3f",444:"12adfae9a2b50e91531f",576:"21a3c1a9ebaebb58d346",703:"17b6b904d18a75a9f3c6",944:"b85c55dff0f14165f872"}[e]+".js?v="+{240:"dd1dbd559d6c95c85b3f",444:"12adfae9a2b50e91531f",576:"21a3c1a9ebaebb58d346",703:"17b6b904d18a75a9f3c6",944:"b85c55dff0f14165f872"}[e],x.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),x.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="@jupyter/docprovider-extension:",x.l=(t,o,a,n)=>{if(e[t])e[t].push(o);else{var i,l;if(void 0!==a)for(var u=document.getElementsByTagName("script"),s=0;s<u.length;s++){var d=u[s];if(d.getAttribute("src")==t||d.getAttribute("data-webpack")==r+a){i=d;break}}i||(l=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,x.nc&&i.setAttribute("nonce",x.nc),i.setAttribute("data-webpack",r+a),i.src=t),e[t]=[o];var f=(r,o)=>{i.onerror=i.onload=null,clearTimeout(p);var a=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),a&&a.forEach((e=>e(o))),r)return r(o)},p=setTimeout(f.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=f.bind(null,i.onerror),i.onload=f.bind(null,i.onload),l&&document.head.appendChild(i)}},x.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{x.S={};var e={},r={};x.I=(t,o)=>{o||(o=[]);var a=r[t];if(a||(a=r[t]={}),!(o.indexOf(a)>=0)){if(o.push(a),e[t])return e[t];x.o(x.S,t)||(x.S[t]={});var n=x.S[t],i="@jupyter/docprovider-extension",l=(e,r,t,o)=>{var a=n[e]=n[e]||{},l=a[r];(!l||!l.loaded&&(!o!=!l.eager?o:i>l.from))&&(a[r]={get:t,from:i,eager:!!o})},u=[];return"default"===t&&(l("@jupyter/collaborative-drive","3.1.0",(()=>Promise.all([x.e(262),x.e(444)]).then((()=>()=>x(444))))),l("@jupyter/docprovider-extension","3.1.0",(()=>Promise.all([x.e(260),x.e(401),x.e(703)]).then((()=>()=>x(703))))),l("@jupyter/docprovider","3.1.0",(()=>Promise.all([x.e(260),x.e(262),x.e(240)]).then((()=>()=>x(240))))),l("y-websocket","1.5.4",(()=>Promise.all([x.e(576),x.e(206)]).then((()=>()=>x(576)))))),e[t]=u.length?Promise.all(u).then((()=>e[t]=1)):1}}})(),(()=>{var e;x.g.importScripts&&(e=x.g.location+"");var r=x.g.document;if(!e&&r&&(r.currentScript&&"SCRIPT"===r.currentScript.tagName.toUpperCase()&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");if(t.length)for(var o=t.length-1;o>-1&&(!e||!/^http(s?):/.test(e));)e=t[o--].src}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),x.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),o=t[1]?r(t[1]):[];return t[2]&&(o.length++,o.push.apply(o,r(t[2]))),t[3]&&(o.push([]),o.push.apply(o,r(t[3]))),o},o=(e,r)=>{e=t(e),r=t(r);for(var o=0;;){if(o>=e.length)return o<r.length&&"u"!=(typeof r[o])[0];var a=e[o],n=(typeof a)[0];if(o>=r.length)return"u"==n;var i=r[o],l=(typeof i)[0];if(n!=l)return"o"==n&&"n"==l||"s"==l||"u"==n;if("o"!=n&&"u"!=n&&a!=i)return a<i;o++}},a=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var o=1,n=1;n<e.length;n++)o--,t+="u"==(typeof(l=e[n]))[0]?"-":(o>0?".":"")+(o=2,l);return t}var i=[];for(n=1;n<e.length;n++){var l=e[n];i.push(0===l?"not("+u()+")":1===l?"("+u()+" || "+u()+")":2===l?i.pop()+" "+i.pop():a(l))}return u();function u(){return i.pop().replace(/^\((.+)\)$/,"$1")}},n=(e,r)=>{if(0 in e){r=t(r);var o=e[0],a=o<0;a&&(o=-o-1);for(var i=0,l=1,u=!0;;l++,i++){var s,d,f=l<e.length?(typeof e[l])[0]:"";if(i>=r.length||"o"==(d=(typeof(s=r[i]))[0]))return!u||("u"==f?l>o&&!a:""==f!=a);if("u"==d){if(!u||"u"!=f)return!1}else if(u)if(f==d)if(l<=o){if(s!=e[l])return!1}else{if(a?s>e[l]:s<e[l])return!1;s!=e[l]&&(u=!1)}else if("s"!=f&&"n"!=f){if(a||l<=o)return!1;u=!1,l--}else{if(l<=o||d<f!=a)return!1;u=!1}else"s"!=f&&"n"!=f&&(u=!1,l--)}}var p=[],c=p.pop.bind(p);for(i=1;i<e.length;i++){var v=e[i];p.push(1==v?c()|c():2==v?c()&c():v?n(v,r):!c())}return!!c()},i=(e,r)=>e&&x.o(e,r),l=e=>(e.loaded=1,e.get()),u=e=>Object.keys(e).reduce(((r,t)=>(e[t].eager&&(r[t]=e[t]),r)),{}),s=(e,r,t,a)=>{var i=a?u(e[r]):e[r];return(r=Object.keys(i).reduce(((e,r)=>!n(t,r)||e&&!o(e,r)?e:r),0))&&i[r]},d=(e,r,t)=>{var a=t?u(e[r]):e[r];return Object.keys(a).reduce(((e,r)=>!e||!a[e].loaded&&o(e,r)?r:e),0)},f=(e,r,t,o)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+a(o)+")",p=(e,r,t,o,n)=>{var i=e[t];return"No satisfying version ("+a(o)+")"+(n?" for eager consumption":"")+" of shared module "+t+" found in shared scope "+r+".\nAvailable versions: "+Object.keys(i).map((e=>e+" from "+i[e].from)).join(", ")},c=e=>{throw new Error(e)},v=e=>{"undefined"!=typeof console&&console.warn&&console.warn(e)},b=(e,r,t)=>t?t():((e,r)=>c("Shared module "+r+" doesn't exist in shared scope "+e))(e,r),y=(h=e=>function(r,t,o,a,n){var i=x.I(r);return i&&i.then&&!o?i.then(e.bind(e,r,x.S[r],t,!1,a,n)):e(r,x.S[r],t,o,a,n)})(((e,r,t,o,a,n)=>{if(!i(r,t))return b(e,t,n);var u=s(r,t,a,o);return u?l(u):n?n():void c(p(r,e,t,a,o))})),m=h(((e,r,t,o,a,u)=>{if(!i(r,t))return b(e,t,u);var s=d(r,t,o);return n(a,s)||v(f(r,t,s,a)),l(r[t][s])})),g={},j={123:()=>m("default","@jupyterlab/apputils",!1,[1,4,4,3]),624:()=>m("default","@jupyterlab/translation",!1,[1,4,3,3]),672:()=>m("default","@jupyterlab/coreutils",!1,[1,6,3,3]),0:()=>m("default","@jupyter/ydoc",!1,[1,3,0,0,,"a3"]),132:()=>m("default","@jupyter/docprovider",!1,[1,3,1,0],(()=>Promise.all([x.e(262),x.e(240)]).then((()=>()=>x(240))))),180:()=>m("default","@jupyterlab/application",!1,[1,4,3,3]),227:()=>m("default","@jupyterlab/settingregistry",!1,[1,4,3,3]),236:()=>m("default","@jupyter/collaborative-drive",!1,[1,3,1,0],(()=>Promise.all([x.e(262),x.e(444)]).then((()=>()=>x(444))))),243:()=>m("default","@jupyterlab/fileeditor",!1,[1,4,3,3]),256:()=>m("default","@lumino/widgets",!1,[1,2,3,1,,"alpha",0]),295:()=>m("default","@jupyterlab/statusbar",!1,[1,4,3,3]),343:()=>m("default","@jupyterlab/logconsole",!1,[1,4,3,3]),920:()=>m("default","@jupyterlab/filebrowser",!1,[1,4,3,3]),977:()=>m("default","@jupyterlab/notebook",!1,[1,4,3,3]),262:()=>m("default","@lumino/coreutils",!1,[1,2,0,0]),345:()=>m("default","react",!1,[1,18,2,0]),560:()=>y("default","y-websocket",!1,[1,1,3,15],(()=>Promise.all([x.e(576),x.e(206)]).then((()=>()=>x(576))))),597:()=>m("default","@jupyterlab/ui-components",!1,[1,4,3,3]),602:()=>m("default","@lumino/signaling",!1,[1,2,0,0]),943:()=>m("default","@jupyterlab/services",!1,[1,7,3,3]),206:()=>m("default","yjs",!1,[1,13,5,40])},w={206:[206],240:[345,560,597,602,943],260:[123,624,672],262:[262],401:[0,132,180,227,236,243,256,295,343,920,977]},P={},x.f.consumes=(e,r)=>{x.o(w,e)&&w[e].forEach((e=>{if(x.o(g,e))return r.push(g[e]);if(!P[e]){var t=r=>{g[e]=0,x.m[e]=t=>{delete x.c[e],t.exports=r()}};P[e]=!0;var o=r=>{delete g[e],x.m[e]=t=>{throw delete x.c[e],r}};try{var a=j[e]();a.then?r.push(g[e]=a.then(t).catch(o)):t(a)}catch(e){o(e)}}}))},(()=>{var e={552:0};x.f.j=(r,t)=>{var o=x.o(e,r)?e[r]:void 0;if(0!==o)if(o)t.push(o[2]);else if(/^(2(06|60|62)|401)$/.test(r))e[r]=0;else{var a=new Promise(((t,a)=>o=e[r]=[t,a]));t.push(o[2]=a);var n=x.p+x.u(r),i=new Error;x.l(n,(t=>{if(x.o(e,r)&&(0!==(o=e[r])&&(e[r]=void 0),o)){var a=t&&("load"===t.type?"missing":t.type),n=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+a+": "+n+")",i.name="ChunkLoadError",i.type=a,i.request=n,o[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var o,a,[n,i,l]=t,u=0;if(n.some((r=>0!==e[r]))){for(o in i)x.o(i,o)&&(x.m[o]=i[o]);l&&l(x)}for(r&&r(t);u<n.length;u++)a=n[u],x.o(e,a)&&e[a]&&e[a][0](),e[a]=0},t=self.webpackChunk_jupyter_docprovider_extension=self.webpackChunk_jupyter_docprovider_extension||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),x.nc=void 0;var E=x(496);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["@jupyter/docprovider-extension"]=E})();