webpackJsonp([0],{"0HId":function(e,t,s){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var a=s("jyVo"),i={data:()=>({failed:!1,username:"",password:"",checked:!1,tips:""}),mounted(){this.username=localStorage.getItem("username"),this.password=localStorage.getItem("password"),this.checked="true"===localStorage.getItem("checked")},methods:{download(){location="/static/client.zip"},login(){if(!this.username||!this.password)return this.tips="请输入用户名和密码",setTimeout(()=>this.tips="",2e3),!1;Object(a.b)("user/login",this.$data).then(e=>{e.code||(localStorage.setItem("username",this.username),localStorage.setItem("checked",this.checked),this.checked?localStorage.setItem("password",this.password):localStorage.setItem("password",""),localStorage.jwt=e.data.jwt,localStorage.removeItem("searchParams"),"sa"==Object(a.o)().role?this.$router.push("/manage"):"check"==Object(a.o)().role?this.$router.push("/check"):this.$router.push("/index"))}).catch(e=>{this.failed=!0,setTimeout(()=>this.failed=!1,2e3)})}}},o={render:function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"login-bg"},[s("div",{staticClass:"tool-tip"},[e._v("客服热线：400-8910357")]),e._v(" "),s("div",{staticClass:"title",on:{click:e.download}}),e._v(" "),s("div",{staticClass:"cnter"}),e._v(" "),s("div",{staticClass:"login"},[e._m(0),e._v(" "),s("div",{staticClass:"name"},[s("input",{directives:[{name:"model",rawName:"v-model",value:e.username,expression:"username"}],attrs:{type:"text",placeholder:"请输入用户名",maxlength:"32",name:"username"},domProps:{value:e.username},on:{keyup:function(t){return!t.type.indexOf("key")&&e._k(t.keyCode,"enter",13,t.key,"Enter")?null:e.login(t)},input:function(t){t.target.composing||(e.username=t.target.value)}}})]),e._v(" "),s("div",{staticClass:"password"},[s("input",{directives:[{name:"model",rawName:"v-model",value:e.password,expression:"password"}],attrs:{type:"text",autocomplete:"off",placeholder:"请输入密码",maxlength:"32",name:"password"},domProps:{value:e.password},on:{keyup:function(t){return!t.type.indexOf("key")&&e._k(t.keyCode,"enter",13,t.key,"Enter")?null:e.login(t)},input:function(t){t.target.composing||(e.password=t.target.value)}}})]),e._v(" "),e.failed&&e.password?s("div",{staticClass:"err"},[e._v("用户名或密码输入错误,请重新输入")]):e._e(),e._v(" "),e.tips?s("div",{staticClass:"err"},[e._v(e._s(e.tips))]):e._e(),e._v(" "),s("div",{staticClass:"rem"},[s("el-checkbox",{model:{value:e.checked,callback:function(t){e.checked=t},expression:"checked"}},[e._v("记住密码")])],1),e._v(" "),s("div",{staticClass:"submit",on:{click:e.login}},[e._v("\n            登录\n        ")])]),e._v(" "),e._m(1)])},staticRenderFns:[function(){var e=this.$createElement,t=this._self._c||e;return t("h1",[this._v("诊断图像处理软件"),t("span",[this._v("dInsight L")])])},function(){var e=this.$createElement,t=this._self._c||e;return t("div",{staticClass:"version"},[t("div",[this._v("版本号：V1.0    完整版本号：V1.0.0.6")]),this._v(" "),t("div",[this._v("杭州迪英加科技有限公司")])])}]};var r=s("VU/8")(i,o,!1,function(e){s("toB5")},"data-v-b2b99062",null);t.default=r.exports},toB5:function(e,t){}});