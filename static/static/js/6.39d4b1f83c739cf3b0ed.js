webpackJsonp([6],{"162o":function(t,e,s){(function(t){var a=void 0!==t&&t||"undefined"!=typeof self&&self||window,i=Function.prototype.apply;function n(t,e){this._id=t,this._clearFn=e}e.setTimeout=function(){return new n(i.call(setTimeout,a,arguments),clearTimeout)},e.setInterval=function(){return new n(i.call(setInterval,a,arguments),clearInterval)},e.clearTimeout=e.clearInterval=function(t){t&&t.close()},n.prototype.unref=n.prototype.ref=function(){},n.prototype.close=function(){this._clearFn.call(a,this._id)},e.enroll=function(t,e){clearTimeout(t._idleTimeoutId),t._idleTimeout=e},e.unenroll=function(t){clearTimeout(t._idleTimeoutId),t._idleTimeout=-1},e._unrefActive=e.active=function(t){clearTimeout(t._idleTimeoutId);var e=t._idleTimeout;e>=0&&(t._idleTimeoutId=setTimeout(function(){t._onTimeout&&t._onTimeout()},e))},s("mypn"),e.setImmediate="undefined"!=typeof self&&self.setImmediate||void 0!==t&&t.setImmediate||this&&this.setImmediate,e.clearImmediate="undefined"!=typeof self&&self.clearImmediate||void 0!==t&&t.clearImmediate||this&&this.clearImmediate}).call(e,s("DuR2"))},"38m7":function(t,e,s){t.exports=s.p+"static/img/add.90d4e5a.svg"},AATC:function(t,e){},Hohg:function(t,e,s){t.exports=s.p+"static/img/delete.468ae5a.svg"},Nm9M:function(t,e){},QxDH:function(t,e,s){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var a=s("jyVo"),i=s("162o"),n={props:["user","type"],data:()=>({alert:"",reg:/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d|\W).{8,32}$/,regs:new RegExp('^[^\\\\\\/:：*?？\\"<>|]+$'),tip:!1}),computed:{userInfo:()=>Object(a.r)()},methods:{addUser(){if(""==this.user.name.trim())return this.alert="用户名不能为空",Object(i.setTimeout)(()=>this.alert="",3e3),!1;if(""==this.user.password&&(this.tip=!0,Object(i.setTimeout)(()=>this.tip=!1,3e3)),this.user.name&&this.user.password&&this.reg.test(this.user.password)){let t;"add"==this.type?(t={grp_name:this.$route.query.data,user_name:this.user.name.trim(),password:this.user.password,role:this.user.userRole,is_test:this.user.isTest?1:0,time_out:this.user.timeOut},Object(a.c)("manage/add_user",t).then(t=>{1==t.data.status?this.$emit("close"):-1==t.data.status&&(this.alert="用户名称已存在",Object(i.setTimeout)(()=>this.alert="",3e3))})):"edit"==this.type&&(t={grp_name:this.$route.query.data,old_user_name:this.user.oldName,id:this.user.id,new_user_name:this.user.name.trim(),password:this.user.password,role:this.user.userRole,is_test:this.user.isTest?1:0,time_out:this.user.timeOut},Object(a.c)("manage/alter_user",t).then(t=>{1==t.data.status?this.$emit("close"):-1==t.data.status&&(this.alert="用户名称已存在",Object(i.setTimeout)(()=>this.alert="",3e3))}))}}}},r={render:function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"modal"},[s("div",{staticClass:"box"},[s("div",{staticClass:"title"},[s("span",[t._v(t._s("add"==t.type?"添加用户":"修改用户"))]),t._v(" "),s("div",{staticClass:"close",on:{click:function(e){return t.$emit("close")}}},[t._v("×")])]),t._v(" "),s("div",{staticClass:"edit"},[s("table",[s("tr",[t._m(0),t._v(" "),s("td",{staticClass:"tips"},[s("input",{directives:[{name:"model",rawName:"v-model",value:t.user.name,expression:"user.name"}],attrs:{maxlength:"32",placeholder:"请输入用户名"},domProps:{value:t.user.name},on:{keyup:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.addModels(e)},input:function(e){e.target.composing||t.$set(t.user,"name",e.target.value)}}}),t._v(" "),s("span",{directives:[{name:"show",rawName:"v-show",value:t.alert,expression:"alert"}]},[t._v(t._s(t.alert))]),t._v(" "),s("span",{directives:[{name:"show",rawName:"v-show",value:t.user.name&&!t.regs.test(t.user.name),expression:"user.name&&!regs.test(user.name)"}]},[t._v('用户名不能包含下列任何字符：\\ / : * ? " < > |')])])]),t._v(" "),s("tr",[t._m(1),t._v(" "),s("td",{staticClass:"tips"},[s("input",{directives:[{name:"model",rawName:"v-model",value:t.user.password,expression:"user.password"}],attrs:{type:"password",placeholder:"请输入密码"},domProps:{value:t.user.password},on:{input:[function(e){e.target.composing||t.$set(t.user,"password",e.target.value)},function(e){t.tip=!1}],keyup:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.addModels(e)}}}),t._v(" "),s("span",{directives:[{name:"show",rawName:"v-show",value:t.user.password&&!t.reg.test(t.user.password),expression:"user.password&&!reg.test(user.password)"}]},[t._v("请输入8-32位,包含大小写字、数字和特殊字符中任意三种组合")]),t._v(" "),s("span",{directives:[{name:"show",rawName:"v-show",value:t.tip,expression:"tip"}]},[t._v("请输入用户密码")])])]),t._v(" "),s("tr",[s("td",{staticStyle:{"padding-top":"5px"},attrs:{align:"right",valign:"top",rowspan:"3"}},[t._v("选择角色")]),t._v(" "),s("td",[s("div",{class:{role:!0,checked:""==t.user.userRole},on:{click:function(e){t.user.userRole=""}}},[t._v("\n                            普通用户\n                        ")]),t._v(" "),"sa"==t.userInfo.role?s("div",{class:{role:!0,checked:"check"==t.user.userRole},on:{click:function(e){t.user.userRole="check"}}},[t._v("                                \n                            复核医生\n                        ")]):t._e(),t._v(" "),s("div",{class:{role:!0,checked:"admin"==t.user.userRole},on:{click:function(e){t.user.userRole="admin"}}},[t._v("\n                            管理员\n                        ")])])])])]),t._v(" "),s("el-button",{on:{click:function(e){return t.$emit("close")}}},[t._v(t._s(t.$t("backstage.cancel")))]),t._v(" "),s("el-button",{attrs:{type:"primary"},on:{click:t.addUser}},[t._v(t._s(t.$t("backstage.confirm")))])],1)])},staticRenderFns:[function(){var t=this.$createElement,e=this._self._c||t;return e("td",{attrs:{align:"right"}},[e("span",{staticStyle:{color:"#FF4B4B"}},[this._v("*")]),this._v("用户名")])},function(){var t=this.$createElement,e=this._self._c||t;return e("td",{attrs:{align:"right"}},[e("span",{staticStyle:{color:"#FF4B4B"}},[this._v("*")]),this._v("用户密码")])}]};var o=s("VU/8")(n,r,!1,function(t){s("zqhU")},"data-v-38d71ceb",null).exports,l={props:["user","type"],data:()=>({titleType:"tct",tableData:[],alert:"",tip:!1,pickTimeValue:""}),computed:{userInfo:()=>Object(a.r)()},mounted(){this.changeTab("tct"),this.initFilterTime()},methods:{initFilterTime(){let t=new Date,e=new Date;e.setTime(e.getTime()-2592e6),this.pickTimeValue=[e,t]},changeTab(t){this.titleType=t,this.initFilterTime(),this.getListData()},getListData(){Object(a.b)("ai/aiStatistics",{aiType:this.titleType,companyid:a.r.company,startTime:this.pickTimeValue&&this.fmtDateToDate(this.pickTimeValue[0]),endTime:this.pickTimeValue&&this.fmtDateToDate(this.pickTimeValue[1])}).then(t=>{this.tableData=t.data})},filterData(){this.getListData()},downloadExcel(t,e){var s=document.createElement("a");s.href="data:text/xlsx;charset=utf-8,\ufeff"+encodeURIComponent(e),s.download=t,s.style.display="none",document.body.appendChild(s),s.click(),document.body.removeChild(s)},exportData(){let t=new Date,e="时间,总用量（例）,阴性,阳性,处理异常".split(","),s=this.tableData.map(t=>[t.date,t.totalCountDr,t.negativeCountDr,t.positiveCountDr,t.abnormalCountDr]);s=[e].concat(s).map(t=>t.map(t=>'"'+(t+"").replace(/"/g,'""')+'"').join(",")).join("\n"),this.downloadExcel(`${this.titleType+t.getTime()}.xlsx`,s)},fmtDateToDate(t){if(!t)return"";var e=t=>(t+100+"").substr(1);return`${(t=new Date(t)).getFullYear()}-${e(t.getMonth()+1)}-${e(t.getDate())}`}}},c={render:function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"modal"},[s("div",{staticClass:"box"},[s("div",{staticClass:"title"},[s("span",[t._v(t._s("用量统计"))]),t._v(" "),s("div",{staticClass:"close",on:{click:function(e){return t.$emit("close")}}},[t._v("×")])]),t._v(" "),s("div",{staticClass:"edit"},[s("div",{staticClass:"tab-title"},[s("li",{staticClass:"title-item",class:{"cur-title":"tct"==t.titleType},on:{click:function(e){return e.stopPropagation(),t.changeTab("tct")}}},[t._v("TCT用量")]),t._v(" "),s("li",{staticClass:"title-item",class:{"cur-title":"lct"==t.titleType},on:{click:function(e){return e.stopPropagation(),t.changeTab("lct")}}},[t._v("LCT用量")]),t._v(" "),s("li",{staticClass:"title-item",class:{"cur-title":"dna"==t.titleType},on:{click:function(e){return e.stopPropagation(),t.changeTab("dna")}}},[t._v("TBS+DNA用量")])]),t._v(" "),s("div",{staticClass:"tab-content"},[s("div",{staticClass:"filter-btn-group"},[s("el-date-picker",{staticClass:"count-picker-time",attrs:{type:"daterange","range-separator":"至","start-placeholder":"开始日期","end-placeholder":"结束日期"},model:{value:t.pickTimeValue,callback:function(e){t.pickTimeValue=e},expression:"pickTimeValue"}}),t._v(" "),s("el-button",{attrs:{type:"primary"},on:{click:function(e){return e.stopPropagation(),t.exportData(e)}}},[t._v("导出")]),t._v(" "),s("el-button",{attrs:{type:"primary"},on:{click:function(e){return e.stopPropagation(),t.filterData(e)}}},[t._v("筛选")])],1),t._v(" "),s("el-table",{key:t.titleType,staticStyle:{width:"100%"},attrs:{data:t.tableData,stripe:""}},[s("el-table-column",{attrs:{prop:"date",label:"月份","min-width":"95"}}),t._v(" "),s("el-table-column",{attrs:{prop:"totalCountDr",label:"总用量（例）","min-width":"125"}}),t._v(" "),s("el-table-column",{attrs:{label:"阴性",prop:"negativeCountDr","min-width":"125"}}),t._v(" "),s("el-table-column",{attrs:{label:"阳性",prop:"positiveCountDr","min-width":"125"}}),t._v(" "),s("el-table-column",{attrs:{prop:"abnormalCount",label:"处理异常","min-width":"105"}})],1)],1)]),t._v(" "),s("el-button",{attrs:{type:"primary"},on:{click:function(e){return t.$emit("close")}}},[t._v(t._s(t.$t("关闭")))])],1)])},staticRenderFns:[]};var u={components:{model:o,usageData:s("VU/8")(l,c,!1,function(t){s("AATC")},"data-v-38c09e6c",null).exports},data:()=>({showModel:!1,showUsageData:!1,userlist:[],user:{oldName:"",name:"",password:"",userRole:"",isTest:!1,timeOut:1e3,id:""}}),methods:{gobackmarmang(){"sa"==Object(a.r)().role&&this.$router.go(-1)},initGetUser(){if(!Object(a.r)().role||"admin"!=Object(a.r)().role&&"sa"!=Object(a.r)().role)this.$router.go(-1);else{var t={grp_name:this.$route.query.data};Object(a.b)("manage/get_users",t).then(t=>{this.userlist=t.data})}},getOneUser(t){this.clearData();let e={grp_name:this.$route.query.data,user_name:t};Object(a.b)("manage/get_user",e).then(t=>{this.user.oldName=t.data.name,this.user.name=t.data.name,this.user.password=t.data.password,this.user.userRole=t.data.role,this.user.timeOut=t.data.time_out,this.user.id=t.data.id,this.user.isTest=1==t.data.is_test,this.showModel="edit"})},clearData(){this.alert="",this.user.oldName="",this.user.name="",this.user.password="",this.user.userRole="",this.user.isTest=!1,this.user.timeOut=1e3,this.user.id=""},deleteuser(t){this.$confirm(this.$t("backstage.ifDel")+t+" ?",{confirmButtonText:this.$t("backstage.confirm"),cancelButtonText:this.$t("backstage.cancel"),type:"warning"}).then(()=>{this.$confirm(this.$t("backstage.confirmdel")+" "+t+" ?",{confirmButtonText:this.$t("backstage.confirm"),cancelButtonText:this.$t("backstage.cancel"),type:"warning"}).then(()=>{let e={grp_name:this.$route.query.data,user_name:t};Object(a.b)("manage/del_user",e).then(t=>{1==t.data.status&&this.initGetUser()}).catch(t=>{console.log(t)})}).catch(()=>{this.$message({type:"info",message:this.$t("backstage.cancelDel")})})}).catch(()=>{this.$message({type:"info",message:this.$t("backstage.cancelDel")})})},close(){this.showModel=!1,this.clearData(),this.initGetUser()}},mounted(){this.initGetUser()}},d={render:function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"admin"},[a("div",{staticClass:"header"},[a("h3",{on:{click:function(e){return t.gobackmarmang()}}},[t._v(t._s(this.$t("backstage.system")))]),t._v(" "),a("router-link",{staticClass:"gohome",attrs:{to:"/index"}},[t._v(t._s(t.$t("backstage.homepage")))])],1),t._v(" "),a("div",{staticClass:"content"},[a("div",{staticClass:"title"},[a("div",{staticClass:"fl"},[t._v("\n                用户信息\n                "),a("span",{staticClass:"num"},[t._v(t._s(t.userlist.length))])]),t._v(" "),a("div",[a("el-button",{attrs:{type:"primary"},on:{click:function(e){t.showUsageData=!0}}},[a("img",{attrs:{src:s("WSF4"),width:"11",alt:""}}),t._v("   "+t._s(t.$t("用量统计")))]),t._v(" "),a("el-button",{attrs:{type:"primary"},on:{click:function(e){t.showModel="add"}}},[a("img",{attrs:{src:s("38m7"),width:"11",alt:""}}),t._v("   "+t._s(t.$t("backstage.addUser")))])],1)]),t._v(" "),a("el-table",{staticClass:"adminlist",staticStyle:{width:"100%"},attrs:{data:t.userlist}},[a("el-table-column",{attrs:{label:this.$t("backstage.username")},scopedSlots:t._u([{key:"default",fn:function(e){return[t._v("\n                    "+t._s(e.row.name)+"\n                ")]}}])}),t._v(" "),a("el-table-column",{attrs:{label:"角色"},scopedSlots:t._u([{key:"default",fn:function(e){return["admin"==e.row.role?a("span",{staticClass:"role manager"},[t._v("管理员")]):"check"==e.row.role?a("span",{staticClass:"role common"},[t._v("复核医生")]):""==e.row.role?a("span",{staticClass:"role common"},[t._v("普通用户")]):t._e()]}}])}),t._v(" "),a("el-table-column",{attrs:{label:t.$t("backstage.operation"),width:"150"},scopedSlots:t._u([{key:"default",fn:function(e){return[a("img",{staticStyle:{cursor:"pointer"},attrs:{src:s("rKNL"),alt:""},on:{click:function(s){return s.stopPropagation(),t.getOneUser(e.row.name)}}}),t._v("  \n                    "),a("img",{staticStyle:{cursor:"pointer"},attrs:{src:s("Hohg"),alt:""},on:{click:function(s){return s.stopPropagation(),t.deleteuser(e.row.name)}}})]}}])})],1)],1),t._v(" "),a("transition",{attrs:{name:"fade"}},[t.showModel?a("model",{attrs:{user:t.user,type:t.showModel},on:{close:t.close}}):t._e()],1),t._v(" "),a("transition",{attrs:{name:"fade"}},[t.showUsageData?a("usageData",{on:{close:function(e){t.showUsageData=!1}}}):t._e()],1)],1)},staticRenderFns:[]};var m=s("VU/8")(u,d,!1,function(t){s("Nm9M"),s("RdXt")},"data-v-510641be",null);e.default=m.exports},RdXt:function(t,e){},WSF4:function(t,e,s){t.exports=s.p+"static/img/yongliangtongji.032f6af.svg"},mypn:function(t,e,s){(function(t,e){!function(t,s){"use strict";if(!t.setImmediate){var a,i,n,r,o,l=1,c={},u=!1,d=t.document,m=Object.getPrototypeOf&&Object.getPrototypeOf(t);m=m&&m.setTimeout?m:t,"[object process]"==={}.toString.call(t.process)?a=function(t){e.nextTick(function(){h(t)})}:!function(){if(t.postMessage&&!t.importScripts){var e=!0,s=t.onmessage;return t.onmessage=function(){e=!1},t.postMessage("","*"),t.onmessage=s,e}}()?t.MessageChannel?((n=new MessageChannel).port1.onmessage=function(t){h(t.data)},a=function(t){n.port2.postMessage(t)}):d&&"onreadystatechange"in d.createElement("script")?(i=d.documentElement,a=function(t){var e=d.createElement("script");e.onreadystatechange=function(){h(t),e.onreadystatechange=null,i.removeChild(e),e=null},i.appendChild(e)}):a=function(t){setTimeout(h,0,t)}:(r="setImmediate$"+Math.random()+"$",o=function(e){e.source===t&&"string"==typeof e.data&&0===e.data.indexOf(r)&&h(+e.data.slice(r.length))},t.addEventListener?t.addEventListener("message",o,!1):t.attachEvent("onmessage",o),a=function(e){t.postMessage(r+e,"*")}),m.setImmediate=function(t){"function"!=typeof t&&(t=new Function(""+t));for(var e=new Array(arguments.length-1),s=0;s<e.length;s++)e[s]=arguments[s+1];var i={callback:t,args:e};return c[l]=i,a(l),l++},m.clearImmediate=p}function p(t){delete c[t]}function h(t){if(u)setTimeout(h,0,t);else{var e=c[t];if(e){u=!0;try{!function(t){var e=t.callback,a=t.args;switch(a.length){case 0:e();break;case 1:e(a[0]);break;case 2:e(a[0],a[1]);break;case 3:e(a[0],a[1],a[2]);break;default:e.apply(s,a)}}(e)}finally{p(t),u=!1}}}}}("undefined"==typeof self?void 0===t?this:t:self)}).call(e,s("DuR2"),s("W2nU"))},rKNL:function(t,e,s){t.exports=s.p+"static/img/edit.4f9db56.svg"},zqhU:function(t,e){}});