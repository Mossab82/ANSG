import os

import xml.etree.ElementTree as ET
import glob
import mysql.connector

class vbo:

    def __init__(self,v):
        self.verb=v
        self.root=None
        self.deverbals = list()
        self.particple=None
        self.NP = list()
        self.Syntax = list()
        self.cat=None
        self.Prep=list()

def DataBase_Connector(DataBase):

    mydb = mysql.connector.connect(user=os.getenv("ANSG_MYSQL_USER", "root"), password=os.environ["ANSG_MYSQL_PASSWORD"],
                              host=os.getenv("ANSG_MYSQL_HOST", "127.0.0.1"), database=DataBase,
                              auth_plugin='mysql_native_password')

    return mydb
if __name__ == '__main__':
    files=glob.glob("mm/*.xml")
    Counter=1
    vbo_l = list()

    for f in files:
        print("<--------Proccising File=",f,"------------>")
        tree = ET.parse(f)
        root=tree.getroot()



        for elem in tree.iter(tag='MEMBERS'):
             for e in elem:
                dev=list()
                verb=e.attrib['transname']
                Participe =None
                for t in e:
                     if(t.tag=="Root"):
                        root=t.attrib['name']
                     if(t.tag=="Deverbal"):
                         dev.append(t.attrib['name'])
                     if (t.tag == "Participle"):
                         Participe= t.attrib['name']


                found_verb=vbo(verb)
                found_verb.root=root
                found_verb.deverbals =dev
                found_verb. particple=Participe
                found_verb.cat=Counter
                vbo_l.append(found_verb)


             for elem in tree.iter(tag='FRAMES'):
                 syn=list()
                 for e in elem:
                        for d in e:
                            if (d.tag == "DESCRIPTION"):
                                if not(d.attrib in syn):
                                   syn.append(d.attrib['primary'])

                 NP=list()
                 prep=list()
                 for e in elem:
                  try:
                     for d in e:
                            if (d.tag == "SYNTAX"):
                                for dd in d:
                                    if (dd.tag=="NP"):
                                        if not dd.attrib['value'] in NP:
                                            NP.append(dd.attrib['value'])
                                    elif(dd.tag=="PREP"):
                                        if not dd.attrib['value'] in prep:
                                            prep.append(dd.attrib['value'])
                  except:
                     print("--------------------------------------------------------->Error")

                               # if not(d.attrib in syn):
                                #   syn.append(d.attrib['value'])
             print("<---------------------" + str(Counter)+ "----------------------->")

        Counter = Counter + 1
print(len(vbo_l))
db = DataBase_Connector("VerbNet")
cursor = db.cursor()


last_countr=0
for v in vbo_l:
    try:

        if (len(v.deverbals) == 0):
            sql = "INSERT INTO Ara_Verbs (Verb,Root,Pariciple,Catagory) VALUES ( %s, %s, %s, %s)"
            val = (v.verb, v.root, v.particple, v.cat)
            cursor.execute(sql, val)
            db.commit()
        if(len(v.deverbals)==1):
            sql = "INSERT INTO Arabic_Verbs (Verb,Root,Deverbal_1,Pariciple,Catagory) VALUES ( %s, %s, %s, %s, %s)"
            val = (v.verb, v.root, v.deverbals[0], v.particple, v.cat)
            cursor.execute(sql, val)
            db.commit()
        elif(len(v.deverbals)==2):
            sql = "INSERT INTO Arabic_Verbs (Verb,Root,Deverbal_1,Deverbal_2,Pariciple,Catagory) VALUES ( %s, %s, %s, %s, %s, %s)"
            val = (v.verb, v.root, v.deverbals[0], v.deverbals[1], v.particple, v.cat)
            cursor.execute(sql, val)
            db.commit()
        elif(len(v.deverbals)==3):
            sql = "INSERT INTO Arabic_Verbs (Verb,Root,Deverbal_1,Deverbal_2,Deverbal_3,Pariciple,Catagory) VALUES ( %s, %s, %s, %s, %s, %s, %s)"
            val = (v.verb, v.root, v.deverbals[0], v.deverbals[1], v.deverbals[2], v.particple, v.cat)
            cursor.execute(sql, val)
            db.commit()
    except:
        last_countr = last_countr + 1
        print("----------------------------------->Database Error")


print("done")
print(last_countr)