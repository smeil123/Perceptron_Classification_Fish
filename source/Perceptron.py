import sys
import os.path as op
import copy
import matplotlib.pyplot as plt
import numpy as np

fd1 = open('salmon_train.txt','r')
salmon_t = fd1.readlines()
fd1.close()
salmon = []

for line in salmon_t:
    a = line.split()
    salmon.append([1,float(a[1]),float(a[0]),1])
        
fd2 = open('seabass_train.txt','r')
seabass_t = fd2.readlines()
fd2.close()
seabass = []

for line in seabass_t:
    a = line.split()
    seabass.append([1,float(a[1]),float(a[0]),0])
        


#error율 계산
def errorEval(weight):
    count = 0
    for i in range(0,50):
        if (weight[2]*salmon[i][2]+weight[1]*salmon[i][1]+weight[0])>0:
            #salmon 분류 실패
            count +=1
        if (weight[2]*seabass[i][2]+weight[1]*seabass[i][1]+weight[0])<0:
            #seabass 분류 실패
            count +=1
    #분류 실패한 수를 세서 반환        
    return count
    
def Perceptron(learningrate):
    
    #2.450267,-1.307926,-179.750256
    #c y x
    pp_weight = [-180,-1,2]

    fdname = 'train_log_%f.txt' % (learningrate)
    fd = open(fdname,'w')
    
    #연어와 농어를 합쳐서 같이 학습한다
    learndata = []
    learndata.extend(salmon)
    learndata.extend(seabass)
    
    error = errorEval(pp_weight)
    print('0 th train error number# : %d' % (error))
    t = 1

    #반복횟수 설정
    for i in range(0,100000):
        o = []
        error = 0
        #연어+농어=100개, 100번 반복
        for j in range(0,100):
            #<weight,x>
            g_eval = (pp_weight[0]*learndata[j][0]+pp_weight[1]*learndata[j][1]+pp_weight[2]*learndata[j][2])
            # 분류자를 기준으로 왼쪽에 있으면 1 오른쪽에 있으면 0 으로 출력해준다
            g_eval = -g_eval
            #0보다 크면 1을 출력
            if(g_eval>0):
                o.append(1)
            #0보다 작거나 같으면 0을 출력
            else:
                o.append(0)
            #출력값을 보고 정답과 다르면 학습
            if(o[j] != learndata[j][3]):
                #틀린 개수를 누적해서 while조건문 확인을 한다
                # weight값을 학습한다
                for k in range(0,3):
                    pp_weight[k] = pp_weight[k] - learningrate*(learndata[j][k]*(learndata[j][3]-o[j]))
            
            error = errorEval(pp_weight)
            if(error<=9):
                #포문탈출
                print('%d th train error number# : %d' % (t,error))                
                fd.write('%d th train error number# : %d\n' % (t,error))
                fd.write('%d th train weight : %f  %f  %f\n' % (t,pp_weight[2],pp_weight[1],pp_weight[0]))
                return pp_weight
                break
                    
        print('%d th train error number# : %d' % (t,error))
        fd.write('%d th train error number# : %d\n' % (t,error))
        fd.write('%d th train weight : %f  %f  %f\n' % (t,pp_weight[2],pp_weight[1],pp_weight[0]))
        t = t+1
    print('train weight =-> %.2f  %.2f  %.2f' % (pp_weight[2],pp_weight[1],pp_weight[0]))
    fd.close()
    return pp_weight
    
def test(weight):
    fd1 = open('salmon_test.txt','r')
    salmon_t = fd1.readlines()
    salmon = []
    fd1.close()
    for lines in salmon_t:
        a = lines.split()
        salmon.append([float(a[0]),float(a[1])])
    
    fd2 = open('seabass_test.txt','r')
    seabass_t = fd2.readlines()
    seabass = []
    fd2.close()
    for lines in seabass_t:
        b = lines.split()
        seabass.append([float(b[0]),float(b[1])])
    

    tmp_x=weight[2]
    tmp_y=weight[1]
    tmp_z=weight[0]
    parameter = [tmp_x,tmp_y,tmp_z]
    x_coefficient = -(parameter[0]/parameter[1])
    x_constant =(parameter[2]/parameter[1])
         
    #출력값 저장하기위한 txt파일 오픈
    f_name = '%s%f%s' %('test_output_',learningrate,'.txt')
    fd3 = open(f_name,'w')
    
    #이 배열들은 그림으로 분류결과를 나타내기 위해 관리한다
    c_salmon = []
    m_salmon = []
    c_seabass = []
    m_seabass = []
    
    #분류시작, 출력 결과를 txt파일에 적는다
    #이때, 그림으로 분류성공과 분류실패를 나타내기위해서 성공했을때와 실패했을때를 분류해서 list에 저장해둔다
    for i in range(0,50):
        if (parameter[0]*salmon[i][0]+parameter[1]*salmon[i][1]+parameter[2])>0:
            #분류실패
            fd3.write('%s%d%s%d%s' %('salmon =',salmon[i][0],', ',salmon[i][1],' =>  fail\n'))
            m_salmon.append(salmon[i])
        else:
            #분류성공
            fd3.write('%s%d%s%d%s' %('salmon =',salmon[i][0],', ',salmon[i][1],' =>  correct\n'))
            c_salmon.append(salmon[i])
            
    for i in range(0,50):        
        if (parameter[0]*seabass[i][0]+parameter[1]*seabass[i][1]+parameter[2])<0:
            #분류실패
            fd3.write('%s%d%s%d%s' %('seabass =',seabass[i][0],', ',seabass[i][1],' =>  fail\n'))
            m_seabass.append(seabass[i])
        else:
            #분류성공
            fd3.write('%s%d%s%d%s' %('seabass =',seabass[i][0],', ',seabass[i][1],' =>  correct\n'))
            c_seabass.append(seabass[i])
            
    #위에서 계산한 결과로 error율을 저장해둔다
    errorrate = (len(m_salmon)+len(m_seabass))*0.01
    fd3.write('%s%f' % ('errorrate => ',errorrate))

    f_name = '%s%f%s' %('test_output_',learningrate,'.png')
    print('%s%f' % ('test errorrate => ',errorrate))
    if __name__ == '__main__':
        fig, ax = plt.subplots()
        
        xList = []
        yList = []
        
        #분류성공한 salmon은 초록색삼각형으로 그린다
        for data in c_salmon:
            x,y = data
            xList.append(x)
            yList.append(y)
        ax.plot(xList,yList,'g^',Label='salmon')
        
        xList = []
        yList = []
        
        #분류성공한 seabass는 노란색사각형으로 그린다
        for data in c_seabass:
            x,y = data
            xList.append(x)
            yList.append(y)
        ax.plot(xList,yList,'ys',Label='seabass')
        
        xList = []
        yList = []
        
        #분류실패한 salmon은 빨간색삼각형으로 그린다
        for data in m_salmon:
            x,y = data
            xList.append(x)
            yList.append(y)
        ax.plot(xList,yList,'r^',Label='salmon')
        
        xList = []
        yList = []
        
        #분류실패한 seabass는 빨간색사각형으로 그린다
        for data in m_seabass:
            x,y = data
            xList.append(x)
            yList.append(y)
        ax.plot(xList,yList,'rs',Label='seabass')
        
        ax.grid(True)
        ax.legend(loc='upper right')
        ax.set_xlabel('Length of body')
        ax.set_ylabel('Length of tail')
        ax.set_xlim((None,None))
        ax.set_ylim((None,None))
        
        #분류자를 빨간색 점선으로 그린다
        a = np.arange(0.0,120.0,0.01)
        ax.plot(a,x_coefficient*a-x_constant,'r--')
        
        plt.savefig(f_name)
        plt.show()
        
    fd3.close()
    
    
def runExp(learningrate):
    print 'training...'
    trResFn = 'train_log__%.2f' % (learningrate)
    weight = Perceptron(learningrate)
    
    print 'result file:',trResFn
    print 'testing...'
    test(weight)
    tsResFn = 'test_output_%.2f' % (learningrate)
    print 'result file:', tsResFn
    
if __name__ == '__main__':
    argmentNum = len(sys.argv)
    
    if argmentNum == 2:
        learningrate = float(sys.argv[1])
        
        runExp(learningrate)
        
    else :
        print('Usage: %s [Learning rate]') % (op.basename(sys.argv[0]))
        