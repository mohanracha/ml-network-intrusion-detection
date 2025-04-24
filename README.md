# Network-Intrusion-Detection-Using-Machine-Learning

## Abstract 

With cyber attacks being a major threat against network and information security, network intrusion detection systems as components of defense-in-depth are indispensable as traditional firewalls alone cannot provide complete protection against intrusion. Intrusion Detection is an important aspect of Network Security and hence an analysis has been conducted on frequently used attacks by applying various Machine Learning techniques. 

Keywords -Network Intrusion, Cyber attacks, Security, Machine Learning.

### Introduction 

Have you ever wondered how your computer/network is able to avoid being infected with malware and bad traffic inputs from the internet? The reason why it can detect it so well is because there are systems in place to protect your valuable information held in your computer or networks. These systems that detect malicious traffic inputs are called Intrusion Detection Systems (IDS) and are trained on internet traffic record data.
An Intrusion Detection System monitors network traffic for suspicious activity and issues an alert when such activity is discovered. 

### Benefits of NIDS

1. NIDS identify and prevent security threats from compromising secure networks.
2. The deployment of NIDS has little impact on network performance. 
3. NIDS are usually passive devices that listen on a network without interfering with the normal operation of a network.
4. NIDS can be made very secure against attack and even made invisible to many attackers.

### Proposed Methodology

The proposed solution makes use of various machine learning techniques to correctly classify and evaluate network based attacks. The NSL-KDD dataset is used for analyzing various types of attacks which can breach the network. Within the data set exists 4 different classes of attacks: 
Denial of Service (DoS), Probe, User to Root(U2R), and Remote to Local (R2L). A brief description of each attack can be seen below:

1. DoS is an attack that tries to shut down traffic flow to and from the target system. The IDS is flooded with an abnormal amount of traffic, which the system can’t handle, and shuts down to protect itself. This prevents normal traffic from visiting a network. An example of this could be an online retailer getting flooded with online orders on a day with a big sale, and because the network can’t handle all the requests, it will shut down preventing paying customers to purchase anything. This is the most common attack in the data set.

2. Probe or surveillance is an attack that tries to get information from a network. The goal here is to act like a thief and steal important information, whether it be personal information about clients or banking information.

3. U2R is an attack that starts off with a normal user account and tries to gain access to the system or network, as a super-user (root). The attacker attempts to exploit the vulnerabilities in a system to gain root privileges/access.

4. R2L is an attack that tries to gain local access to a remote machine. An attacker does not have local access to the system/network, and tries to “hack” their way into the network.

The current proposition detects various network intrusion based attacks using machine learning techniques. Detection is done by first mapping the attack field to the attack class using the NSL-KDD dataset as a data source. After categorizing each attack field into separate categories, a statistical distribution is inferred and random sampling is done for selecting important features for classifying the type of network intrusion attack. A multimodal approach is then utilized for generating a classification report for evaluation. 

## Technology used

Dataset - NSL-KDD dataset; 
Libraries - matplotlib, pandas, sklearn, numpy, seaborn, imblearn.

## Observations

![image](https://user-images.githubusercontent.com/79298507/134902123-e6bc1108-83cd-4f7d-8d93-cb68708da1d6.png)
![image](https://user-images.githubusercontent.com/79298507/134902242-6c3f5fb5-18e6-48ff-804d-312f15a97deb.png)

### Model accuracy after evaluating the testing models

![image](https://user-images.githubusercontent.com/79298507/134902854-428bb330-edef-4f78-859f-a1e58d2de5aa.png)
![image](https://user-images.githubusercontent.com/79298507/134902905-8dae6551-a201-4bcc-80f2-784ea58509be.png)
![image](https://user-images.githubusercontent.com/79298507/134902956-f6dfda35-a1ed-4191-8299-b1436a30a2da.png)
![image](https://user-images.githubusercontent.com/79298507/134903032-cf142028-6d24-4137-a4a1-4a7b308f4b4a.png)

## Future Enhancement

The current proposition has a limited scope of detecting various types of network intrusion based attacks. The attacks can be categorized into each attack class which is present in the NSL-KDD dataset, any different type of intrusive approach may not be detected. 

## References

[1] https://www.sciencedirect.com/topics/computer-science/network-based-intrusion-detection-system

[2] https://www.researchgate.net/publication/45681663_Research_on_Intrusion_Detection_and_Response_A_Survey

[3] Ahmad, Z, Shahid Khan, A, Wai Shiang, C, Abdullah, J, Ahmad, F. Network intrusion detection system: A systematic study of machine learning and deep learning approaches. Trans Emerging Tel Tech. 2021. https://doi.org/10.1002/ett.4150

[4] https://www.unb.ca/cic/datasets/nsl.html




