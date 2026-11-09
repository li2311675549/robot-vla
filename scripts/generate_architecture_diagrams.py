from pathlib import Path
from html import escape
import argparse
import math

FONT = 'Noto Sans CJK SC,Microsoft YaHei,sans-serif'
COLORS = {'blue':('#eef5ff','#3065a0'), 'green':('#edf8f4','#26735c'), 'orange':('#fff6ea','#a56b20'), 'purple':('#f3effc','#7353a2'), 'gray':('#f4f6f9','#536579')}
class Diagram:
    def __init__(self, width, height, title, description):
        self.w,self.h=width,height
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>', f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
        self.text(32,38,title,26,bold=True)
        self.nodes={}
    def text(self,x,y,value,size=18,color='#21364d',bold=False,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
    def box(self,key,x,y,w,h,title,lines=(),tone='blue',size=18):
        fill,stroke=COLORS[tone]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.text(x+18,y+29,title,20,stroke,True)
        for i,line in enumerate(lines):self.text(x+18,y+56+24*i,line,size)
        self.nodes[key]=(x,y,w,h)
    def line(self,points,dashed=False):
        pts=' '.join(f'{x},{y}' for x,y in points)
        dash=' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="#63758a" stroke-width="1.8"{dash}/>')
        # Explicit triangles render consistently in GitHub and offline SVG viewers.
        x,y=points[-1];px,py=points[-2];length=math.hypot(x-px,y-py)
        ux,uy=(x-px)/length,(y-py)/length
        bx,by=x-10*ux,y-10*uy
        triangle=f'{x},{y} {bx-4.5*uy},{by+4.5*ux} {bx+4.5*uy},{by-4.5*ux}'
        self.parts.append(f'<polygon points="{triangle}" fill="#63758a"/>')
    def down(self,a,b):
        x,y,w,h=self.nodes[a];xx,yy,ww,hh=self.nodes[b]
        assert x+w/2==xx+ww/2
        self.line([(x+w/2,y+h),(xx+ww/2,yy)])
    def save(self,path):
        path.write_text('\n'.join(self.parts+['</svg>']))

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    d=Diagram(1140,1040,'图1｜传统系统：完整任务如何形成闭环','仅描述传统控制。任务步骤调度器按观测选择导航、驻车操作或门轮臂协同，经真实执行器动作和物理反馈验证后推进；异常有界恢复。VLA独立见图3。')
    d.text(32,71,'范围：5m接近、操作、搬运、放置与任务终检；三项完整流程见图2。',18)
    d.box('input',160,100,820,103,'① 读取任务、场景与当前反馈', ['目标：开关门 / 十果分类 / 三书收纳返航','输入：相机、实际机器人状态；门/抽屉/大书专家另用仿真几何（Oracle）'])
    d.box('schedule',160,233,820,103,'② 任务步骤调度器：判断当前做到哪一步', ['保存目标与进度；选择下一技能；设置完成条件、超时与重试上限','只有真实反馈满足该步条件，才推进下一步；不是一次发完所有动作'])
    d.down('input','schedule')
    d.line([(570,336),(570,359),(207,359),(207,380)])
    d.line([(570,336),(570,380)])
    d.line([(570,359),(933,359),(933,380)])
    d.box('nav',44,380,326,226,'③A 需要换位置：底盘导航', ['已知障碍地图 → 膨胀安全边界','A*搜索 → 路点跟踪','左右轮速 → 四轮真实驱动','终端停靠（Dock）→ 复核稳态','收臂行驶 / 带载时监控夹持'],size=17)
    d.box('arm',407,380,326,226,'③B 到操作区：精细操作', ['感知目标 → 候选抓点/放置点','筛选底盘操作位姿 → IK求解','整套持物碰撞检查 / 关节路径','停稳后推、抓、抬升、放置','限速控制；按配置启用驻车'],tone='green',size=17)
    d.box('door',770,380,326,226,'③C 门操作：底盘与臂协同', ['门角决定把手的圆弧目标','底盘沿圆弧切向跟随','机械臂按实测状态重算目标','接触、跟踪均合格才前进','失稳则暂停 / 安全释放重抓'],tone='orange',size=17)
    d.line([(207,606),(207,630),(570,630),(570,650)])
    d.line([(570,606),(570,650)])
    d.line([(933,606),(933,630),(570,630)])
    d.box('physics',160,650,820,103,'④ 正常执行器 + MuJoCo物理接触', ['轮速、六关节与夹爪命令经控制器执行；物体、门和抽屉由真实接触运动','每2ms推进物理；读取图像、关节、底盘、接触和物体状态'],tone='gray')
    d.box('verify',160,783,820,103,'⑤ 验证当前步骤，再决定继续、恢复或结束', ['正常：进入下一步骤；任务末尾：独立检查门 / 十果 / 三书与返航结果','异常：停车或停止机械臂 → 按技能有限重试；超限明确失败并保存日志'],tone='purple')
    d.down('physics','verify')
    d.line([(160,835),(18,835),(18,285),(160,285)],True)
    d.text(32,949,'虚线＝把实际执行结果交回步骤调度器；不是无条件重复动作。',18)
    d.text(32,981,'相机反馈、接触监控、任务评分的用途不同；评分用真值，不代表视觉策略可读取真值。',18)
    d.text(32,1013,'当前VLA短技能不在此完整任务执行链内；它的独立输入、输出和终点见图3。',18,'#7353a2',True)
    d.save(out/'traditional_control_flow.svg')

    d=Diagram(1200,1300,'图2｜三个完整任务：真实动作的先后顺序','三个并列的传统任务流程。从约5米起点开始；水果循环到十件正确入筐；书本先开抽屉再回桌抓书，逐层完成后返航。门和抽屉、大书使用Oracle基线，水果使用RGB-D与仿真辅助验证。')
    d.text(32,72,'按列自上而下阅读；三列是三个独立任务，不是机器人依次串行做三项。',18)
    routes=[('door','开门 → 入室 → 关门','Oracle几何专家；真实门把手接触','orange',[
        ('① 门外约5m，门初始关闭',['收拢机械臂，保存任务初态']),
        ('② 导航至门外操作区',['规划避障路径，实际停靠复核']),
        ('③ 定位外把手并接近',['检查可达性，闭合并确认接触']),
        ('④ 夹紧把手，轮臂配合拉门',['沿门轴圆弧，按实际门角反馈']),
        ('⑤ 达到可通行开度后退手',['松开把手，收臂，规划过门路径']),
        ('⑥ 整机通过门洞',['检查底盘、机械臂和夹爪净空']),
        ('⑦ 房间内重新接近内把手',['重新停靠、抓握，确认接触']),
        ('⑧ 轮臂配合关门并安全退手',['沿约束轨迹闭合，释放后再复核']),
        ('⑨ 终检：整机在内、门已关好',['位姿与速度合格，记录完整结果'])]),
      ('fruit','十果分类 → 正确入筐','RGB-D生成目标；仿真辅助验证','green',[
        ('① 距桌约5m，桌面随机10果',['苹果、香蕉、橘子均有且有差异']),
        ('② 导航至桌前，观察剩余水果',['颜色分割、深度反投影、主轴估计']),
        ('③ 选择目标与抓取候选',['检查开口、可达性、抓放路径']),
        ('④ 到抓点上方，再下降闭合',['检查左右指接触，抬升后复核']),
        ('⑤ 稳定夹持，实际带载转运',['底盘驶向对应筐，监控物体滑移']),
        ('⑥ 筐前停靠，复算放置目标',['使用实际停车位姿，检查净空']),
        ('⑦ 降低、松爪、退手',['独立检查水果是否稳定落入正确筐']),
        ('⑧ 更新已分类对象与剩余目标',['返回桌面并重新观察，再处理下一件']),
        ('⑨ 终检：10个全部正确入筐',['有剩余、放错或超时均不能算完成'])]),
      ('book','三书分层 → 关闭 → 返航','当前推边侧抓完整基线：Oracle','blue',[
        ('① 距桌约5m，保存返航起点',['三本大小明显不同的书随机散落']),
        ('② 接近书桌，按长×宽排序',['小 / 中 / 大 → 第一 / 二 / 三层']),
        ('③ 空爪先去打开对应抽屉',['抓把手沿滑轨拉开，确认实际开度']),
        ('④ 回书桌，复核当前书姿态',['真实推至桌边，保留支撑及下指空间']),
        ('⑤ 从侧面夹住书边的厚度',['近边向内15mm，夹紧、抬升、复核']),
        ('⑥ 持书前往柜前预停靠位',['检查整套持物净空与抽屉实际开度']),
        ('⑦ 沿检查过的路径入屉放书',['松开、退手，确认书在对应层内']),
        ('⑧ 空爪关闭该层抽屉',['确认关好，再处理下一本']),
        ('⑨ 三本完成后返回保存的起点',['独立检查三书、三屉和返航状态'])])]
    for col,(key,title,subtitle,tone,steps) in enumerate(routes):
        x=26+col*390;w=368
        d.text(x+12,116,title,23,COLORS[tone][1],True)
        d.text(x+12,146,subtitle,16)
        for i,(title,lines) in enumerate(steps):
            node=f'{key}{i}';d.box(node,x,170+i*108,w,84,title,lines,tone,size=17)
            if i:d.down(f'{key}{i-1}',node)
        footer={'door':['接触丢失 / 跟踪异常：暂停并有限重抓','过门或终检不通过：记录失败'], 'fruit':['②–⑧循环至全部10个分类完成','单件成功不等于完整十果成功'], 'book':['③–⑧按小、中、大循环三次','先开屉再取书，避免夹书时操作把手']}[key]
        d.box(key+'footer',x,1160,w,102,'循环与异常处理',footer,'gray',size=16)
    d.save(out/'complete_task_routes.svg')

    d=Diagram(1080,1130,'图3｜当前VLA实验：驻车、按指令抓起并保持','与完整任务独立的短技能评测。初始化为已停车十果桌面，VLA只输入前方及腕部RGB、实际机械臂状态和语言，预测六轴绝对关节目标与夹爪，不输出底盘动作，不搬到筐，不放置。执行一小步重新观测。')
    d.text(32,73,'这是独立的短技能入口；没有5m导航、带载驶向收纳筐或入筐放置。',18,'#7353a2',True)
    items=[('init','① 初始化已驻车的十果桌面场景',['读取评测初态；水果起初未被抓住；驻车稳定检查通过后开始'],'gray'),
      ('observe','② 采集本次观测',['前方RGB + 腕部RGB + 六关节/夹爪实际状态 + 语言指令 + 时间戳'],'purple'),
      ('infer','③ π₀.₅模型预测动作块',['读懂“抓起指定类别的最左水果并保持”；预测连续10步关节/夹爪目标'],'purple'),
      ('guard','④ 解码第1步目标，施加安全约束',['六关节绝对目标 + 夹爪；检查限位，插值并约束速度、加速度'],'blue'),
      ('step','⑤ 正常执行动作，推进0.2秒物理仿真',['机械臂、夹爪真实动作；独立轮速伺服保持驻车，模型不输出轮速'],'blue'),
      ('check','⑥ 持续监控，并独立判定本次短技能是否完成',['目标抬升≥55mm；最近完整1秒≥90%双侧接触，末帧仍有双侧接触','同时检查未抓错目标、其他水果状态与驻车；异常或超时记录失败'],'green')]
    for i,(key,title,lines,tone) in enumerate(items):
        d.box(key,180,108+i*142,860,112,title,lines,tone,size=17)
        if i:d.down(items[i-1][0],key)
    d.line([(180,873),(104,873),(104,306),(180,306)],True)
    d.text(32,956,'虚线＝尚未成功且未触发失败时，再观测、再预测；不是把10步预测一次性全部执行。',18)
    d.text(32,990,'观察 → 预测 → 执行0.2秒 → 再观察，按5Hz仿真节拍闭环；网络等待时仿真暂停。',18)
    d.box('boundary',32,1020,1008,84,'实验终点：指定水果抓起并保持', ['不接入当前完整十果调度循环；入筐与底盘导航没有在本实验中交给模型。'],'orange',size=18)
    d.save(out/'vla_stationary_skill_flow.svg')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Regenerate documentation SVG flowcharts without third-party dependencies.')
    parser.add_argument('--output',type=Path,default=Path('docs/media'))
    main(parser.parse_args().output)
