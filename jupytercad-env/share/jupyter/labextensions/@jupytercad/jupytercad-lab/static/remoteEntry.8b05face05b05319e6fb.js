var _JUPYTERLAB;(()=>{"use strict";var e,r,t,a,n,o,i,u,l,d,p,f,s,c,h,v,b,y,g,m,j,w={625:(e,r,t)=>{var a={"./index":()=>t.e(484).then((()=>()=>t(484))),"./extension":()=>t.e(484).then((()=>()=>t(484))),"./style":()=>t.e(432).then((()=>()=>t(432)))},n=(e,r)=>(t.R=r,r=t.o(a,e)?a[e]():Promise.resolve().then((()=>{throw new Error('Module "'+e+'" does not exist in container.')})),t.R=void 0,r),o=(e,r)=>{if(t.S){var a="default",n=t.S[a];if(n&&n!==e)throw new Error("Container initialization failed as it has already been initialized with a different share scope");return t.S[a]=e,t.I(a,r)}};t.d(r,{get:()=>n,init:()=>o})}},S={};function E(e){var r=S[e];if(void 0!==r)return r.exports;var t=S[e]={id:e,exports:{}};return w[e](t,t.exports,E),t.exports}E.m=w,E.c=S,E.n=e=>{var r=e&&e.__esModule?()=>e.default:()=>e;return E.d(r,{a:r}),r},E.d=(e,r)=>{for(var t in r)E.o(r,t)&&!E.o(e,t)&&Object.defineProperty(e,t,{enumerable:!0,get:r[t]})},E.f={},E.e=e=>Promise.all(Object.keys(E.f).reduce(((r,t)=>(E.f[t](e,r),r)),[])),E.u=e=>e+"."+{432:"8c393580a710cafd1dd0",484:"9d783b6e3c4b2f2b63ef"}[e]+".js?v="+{432:"8c393580a710cafd1dd0",484:"9d783b6e3c4b2f2b63ef"}[e],E.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),E.o=(e,r)=>Object.prototype.hasOwnProperty.call(e,r),e={},r="@jupytercad/jupytercad-lab:",E.l=(t,a,n,o)=>{if(e[t])e[t].push(a);else{var i,u;if(void 0!==n)for(var l=document.getElementsByTagName("script"),d=0;d<l.length;d++){var p=l[d];if(p.getAttribute("src")==t||p.getAttribute("data-webpack")==r+n){i=p;break}}i||(u=!0,(i=document.createElement("script")).charset="utf-8",i.timeout=120,E.nc&&i.setAttribute("nonce",E.nc),i.setAttribute("data-webpack",r+n),i.src=t),e[t]=[a];var f=(r,a)=>{i.onerror=i.onload=null,clearTimeout(s);var n=e[t];if(delete e[t],i.parentNode&&i.parentNode.removeChild(i),n&&n.forEach((e=>e(a))),r)return r(a)},s=setTimeout(f.bind(null,void 0,{type:"timeout",target:i}),12e4);i.onerror=f.bind(null,i.onerror),i.onload=f.bind(null,i.onload),u&&document.head.appendChild(i)}},E.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},(()=>{E.S={};var e={},r={};E.I=(t,a)=>{a||(a=[]);var n=r[t];if(n||(n=r[t]={}),!(a.indexOf(n)>=0)){if(a.push(n),e[t])return e[t];E.o(E.S,t)||(E.S[t]={});var o=E.S[t],i="@jupytercad/jupytercad-lab",u=[];return"default"===t&&((e,r,t,a)=>{var n=o[e]=o[e]||{},u=n[r];(!u||!u.loaded&&(1!=!u.eager?a:i>u.from))&&(n[r]={get:()=>E.e(484).then((()=>()=>E(484))),from:i,eager:!1})})("@jupytercad/jupytercad-lab","3.0.1"),e[t]=u.length?Promise.all(u).then((()=>e[t]=1)):1}}})(),(()=>{var e;E.g.importScripts&&(e=E.g.location+"");var r=E.g.document;if(!e&&r&&(r.currentScript&&"SCRIPT"===r.currentScript.tagName.toUpperCase()&&(e=r.currentScript.src),!e)){var t=r.getElementsByTagName("script");if(t.length)for(var a=t.length-1;a>-1&&(!e||!/^http(s?):/.test(e));)e=t[a--].src}if(!e)throw new Error("Automatic publicPath is not supported in this browser");e=e.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),E.p=e})(),t=e=>{var r=e=>e.split(".").map((e=>+e==e?+e:e)),t=/^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),a=t[1]?r(t[1]):[];return t[2]&&(a.length++,a.push.apply(a,r(t[2]))),t[3]&&(a.push([]),a.push.apply(a,r(t[3]))),a},a=(e,r)=>{e=t(e),r=t(r);for(var a=0;;){if(a>=e.length)return a<r.length&&"u"!=(typeof r[a])[0];var n=e[a],o=(typeof n)[0];if(a>=r.length)return"u"==o;var i=r[a],u=(typeof i)[0];if(o!=u)return"o"==o&&"n"==u||"s"==u||"u"==o;if("o"!=o&&"u"!=o&&n!=i)return n<i;a++}},n=e=>{var r=e[0],t="";if(1===e.length)return"*";if(r+.5){t+=0==r?">=":-1==r?"<":1==r?"^":2==r?"~":r>0?"=":"!=";for(var a=1,o=1;o<e.length;o++)a--,t+="u"==(typeof(u=e[o]))[0]?"-":(a>0?".":"")+(a=2,u);return t}var i=[];for(o=1;o<e.length;o++){var u=e[o];i.push(0===u?"not("+l()+")":1===u?"("+l()+" || "+l()+")":2===u?i.pop()+" "+i.pop():n(u))}return l();function l(){return i.pop().replace(/^\((.+)\)$/,"$1")}},o=(e,r)=>{if(0 in e){r=t(r);var a=e[0],n=a<0;n&&(a=-a-1);for(var i=0,u=1,l=!0;;u++,i++){var d,p,f=u<e.length?(typeof e[u])[0]:"";if(i>=r.length||"o"==(p=(typeof(d=r[i]))[0]))return!l||("u"==f?u>a&&!n:""==f!=n);if("u"==p){if(!l||"u"!=f)return!1}else if(l)if(f==p)if(u<=a){if(d!=e[u])return!1}else{if(n?d>e[u]:d<e[u])return!1;d!=e[u]&&(l=!1)}else if("s"!=f&&"n"!=f){if(n||u<=a)return!1;l=!1,u--}else{if(u<=a||p<f!=n)return!1;l=!1}else"s"!=f&&"n"!=f&&(l=!1,u--)}}var s=[],c=s.pop.bind(s);for(i=1;i<e.length;i++){var h=e[i];s.push(1==h?c()|c():2==h?c()&c():h?o(h,r):!c())}return!!c()},i=(e,r)=>e&&E.o(e,r),u=e=>(e.loaded=1,e.get()),l=e=>Object.keys(e).reduce(((r,t)=>(e[t].eager&&(r[t]=e[t]),r)),{}),d=(e,r,t)=>{var n=t?l(e[r]):e[r];return Object.keys(n).reduce(((e,r)=>!e||!n[e].loaded&&a(e,r)?r:e),0)},p=(e,r,t,a)=>"Unsatisfied version "+t+" from "+(t&&e[r][t].from)+" of shared singleton module "+r+" (required "+n(a)+")",f=e=>{throw new Error(e)},s=e=>{"undefined"!=typeof console&&console.warn&&console.warn(e)},h=(e,r,t)=>t?t():((e,r)=>f("Shared module "+r+" doesn't exist in shared scope "+e))(e,r),v=(c=e=>function(r,t,a,n,o){var i=E.I(r);return i&&i.then&&!a?i.then(e.bind(e,r,E.S[r],t,!1,n,o)):e(r,E.S[r],t,a,n,o)})(((e,r,t,a,n)=>{if(!i(r,t))return h(e,t,n);var o=d(r,t,a);return u(r[t][o])})),b=c(((e,r,t,a,n,l)=>{if(!i(r,t))return h(e,t,l);var f=d(r,t,a);return o(n,f)||s(p(r,t,f,n)),u(r[t][f])})),y={},g={161:()=>b("default","@jupyterlab/application",!1,[1,4,3,4]),200:()=>v("default","@jupyter/docprovider",!1),230:()=>b("default","@lumino/messaging",!1,[1,2,0,0]),256:()=>b("default","@lumino/widgets",!1,[1,2,3,1,,"alpha",0]),282:()=>b("default","@jupytercad/schema",!1,[1,3,0,1]),458:()=>b("default","@jupyterlab/translation",!1,[1,4,3,4]),498:()=>b("default","@jupyter/collaborative-drive",!1,[1,3,1,0,,"alpha",0]),589:()=>b("default","@jupyterlab/mainmenu",!1,[1,4,3,4]),824:()=>b("default","@jupytercad/base",!1,[1,3,0,1]),862:()=>b("default","@jupyterlab/completer",!1,[1,4,3,4]),955:()=>b("default","yjs-widgets",!1,[2,0,3,7])},m={484:[161,200,230,256,282,458,498,589,824,862,955]},j={},E.f.consumes=(e,r)=>{E.o(m,e)&&m[e].forEach((e=>{if(E.o(y,e))return r.push(y[e]);if(!j[e]){var t=r=>{y[e]=0,E.m[e]=t=>{delete E.c[e],t.exports=r()}};j[e]=!0;var a=r=>{delete y[e],E.m[e]=t=>{throw delete E.c[e],r}};try{var n=g[e]();n.then?r.push(y[e]=n.then(t).catch(a)):t(n)}catch(e){a(e)}}}))},(()=>{var e={588:0};E.f.j=(r,t)=>{var a=E.o(e,r)?e[r]:void 0;if(0!==a)if(a)t.push(a[2]);else{var n=new Promise(((t,n)=>a=e[r]=[t,n]));t.push(a[2]=n);var o=E.p+E.u(r),i=new Error;E.l(o,(t=>{if(E.o(e,r)&&(0!==(a=e[r])&&(e[r]=void 0),a)){var n=t&&("load"===t.type?"missing":t.type),o=t&&t.target&&t.target.src;i.message="Loading chunk "+r+" failed.\n("+n+": "+o+")",i.name="ChunkLoadError",i.type=n,i.request=o,a[1](i)}}),"chunk-"+r,r)}};var r=(r,t)=>{var a,n,[o,i,u]=t,l=0;if(o.some((r=>0!==e[r]))){for(a in i)E.o(i,a)&&(E.m[a]=i[a]);u&&u(E)}for(r&&r(t);l<o.length;l++)n=o[l],E.o(e,n)&&e[n]&&e[n][0](),e[n]=0},t=self.webpackChunk_jupytercad_jupytercad_lab=self.webpackChunk_jupytercad_jupytercad_lab||[];t.forEach(r.bind(null,0)),t.push=r.bind(null,t.push.bind(t))})(),E.nc=void 0;var P=E(625);(_JUPYTERLAB=void 0===_JUPYTERLAB?{}:_JUPYTERLAB)["@jupytercad/jupytercad-lab"]=P})();